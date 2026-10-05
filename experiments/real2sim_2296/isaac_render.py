#!/usr/bin/env python3
"""Render an existing USD camera with Isaac Sim; never save edits to the scene.

Run with an installed Isaac Sim Python environment. Example:
  python isaac_render.py --scene scene.usda --camera /World/Camera \
      --output image.png --width 640 --height 480 --gpu 4 --frames 64

--hidden-control also renders with all NuRec volumes hidden in a session layer,
providing pixel evidence that the Gaussian asset contributes to the image.
Use an external process-group timeout for a bounded unattended render.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progress(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def render_settings(stage):
    """Root opinions win; conflicting referenced defaults require a root choice."""
    root = stage.GetRootLayer()
    root_values = dict(root.customLayerData.get("renderSettings", {}))
    merged, sources = {}, []
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: item.identifier):
        values = dict(layer.customLayerData.get("renderSettings", {}))
        if not values or layer == root:
            continue
        sources.append({"layer": layer.identifier, "settings": values})
        for key, value in values.items():
            if key in merged and merged[key] != value and key not in root_values:
                raise ValueError(f"Conflicting referenced render setting {key}; author a canonical root value")
            merged[key] = value
    if root_values:
        sources.append({"layer": root.identifier, "settings": root_values, "precedence": "root"})
    merged.update(root_values)
    return merged, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera", required=True, help="Existing absolute USD Camera prim path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--gpu", type=int, default=0, help="RTX and CUDA device ordinal; inspect startup PCI table")
    parser.add_argument("--frames", type=int, default=64, help="Warm-up render frames for each image")
    parser.add_argument("--subframes", type=int, default=1, help="Replicator subframes per final capture")
    parser.add_argument("--anti-aliasing", type=int, choices=range(5), default=3,
                        help="Isaac AA mode: 0 off, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA")
    parser.add_argument("--hidden-control", action="store_true")
    args = parser.parse_args()
    # SimulationApp forwards unknown sys.argv tokens to Kit; our arguments are
    # already parsed and must not be interpreted again by the application.
    sys.argv = [sys.argv[0]]
    if min(args.width, args.height, args.frames, args.subframes) < 1 or args.gpu < 0:
        parser.error("width, height and frames must be positive, and gpu nonnegative")
    if not args.camera.startswith("/") or args.output.suffix.lower() != ".png":
        parser.error("camera must be an absolute USD prim path and output must be a PNG")
    source = args.scene.resolve(strict=True)
    output = args.output.resolve()
    receipt_path = output.with_suffix(".png.receipt.json")
    status_path = output.with_suffix(".png.status.json")
    control_path = output.with_name(output.stem + "_hidden_control.png")
    for path in (output, receipt_path, status_path, control_path if args.hidden_control else output):
        if path.exists():
            raise ValueError(f"Output already exists: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(source)
    started = time.monotonic()
    receipt = {
        "schema": "real2sim-isaac-render/v1", "produced_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": before},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "request": vars(args) | {"scene": str(source), "output": str(output)},
        "pid": os.getpid(), "status": "started",
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
    }
    app = None
    annotator = product = None

    def snapshot(state):
        receipt["status"] = state
        temporary = status_path.with_suffix(status_path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(temporary, status_path)

    snapshot("starting")
    try:
        # All Kit-dependent imports must follow SimulationApp initialization.
        from isaacsim import SimulationApp

        config = {"headless": True, "hide_ui": True, "active_gpu": args.gpu,
                  "physics_gpu": args.gpu, "multi_gpu": False,
                  "width": args.width, "height": args.height,
                  "renderer": "RaytracedLighting", "sync_loads": True,
                  "disable_viewport_updates": True, "anti_aliasing": args.anti_aliasing,
                  "fast_shutdown": True, "enable_crashreporter": False,
                  "extra_args": ["--portable-root", str(output.parent / "kit-portable"),
                                 "--/app/settings/persistent=false", "--/app/telemetry/enabled=false",
                                 f"--/log/file={output.parent / (output.stem + '.kit.log')}"]}
        progress("initializing_isaac", config=config)
        app = SimulationApp(config)

        import carb.settings
        import numpy as np
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Usd, UsdGeom

        settings = carb.settings.get_settings()
        manager = omni.kit.app.get_app().get_extension_manager()
        schema_name = "omni.usd.schema.omni_nurec_types"
        enabled_id = manager.get_enabled_extension_id(schema_name)
        extension_state = {"name": schema_name, "initially_enabled": enabled_id or None}
        if not enabled_id:
            candidates = [entry for entry in manager.get_extensions()
                          if entry.get("id", "").startswith(schema_name + "-")
                          or entry.get("name") == schema_name]
            extension_state["available_local_ids"] = [entry.get("id") for entry in candidates]
            if candidates:
                extension_state["enable_attempt_result"] = bool(
                    manager.set_extension_enabled_immediate(schema_name, True))
            else:
                extension_state["enable_attempt"] = "not_available_locally; no download or installation attempted"
        extension_state["finally_enabled"] = manager.get_enabled_extension_id(schema_name) or None
        receipt["nurec_schema_extension"] = extension_state
        progress("isaac_initialized", elapsed_seconds=time.monotonic() - started, schema=extension_state)

        context = omni.usd.get_context()
        if not context.open_stage(str(source)):
            raise RuntimeError(f"Isaac could not open {source}")
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac returned no opened stage")
        stage.SetEditTarget(stage.GetSessionLayer())
        camera_prim = stage.GetPrimAtPath(args.camera)
        if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
            raise ValueError(f"Requested Camera prim does not exist: {args.camera}")
        camera = UsdGeom.Camera(camera_prim)
        receipt["camera"] = {
            "path": args.camera,
            "world_transform": np.asarray(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())).tolist(),
            "focal_length": camera.GetFocalLengthAttr().Get(),
            "horizontal_aperture": camera.GetHorizontalApertureAttr().Get(),
            "vertical_aperture": camera.GetVerticalApertureAttr().Get(),
            "horizontal_aperture_offset": camera.GetHorizontalApertureOffsetAttr().Get(),
            "vertical_aperture_offset": camera.GetVerticalApertureOffsetAttr().Get(),
            "clipping_range": list(camera.GetClippingRangeAttr().Get()),
        }
        volumes = [prim for prim in stage.Traverse()
                   if prim.GetAttribute("omni:nurec:isNuRecVolume").Get() is True]
        receipt["nurec_volumes"] = [str(prim.GetPath()) for prim in volumes]
        receipt["nurec_schema_registered"] = bool(
            Usd.SchemaRegistry().FindConcretePrimDefinition("OmniNuRecFieldAsset"))
        requested_settings, settings_sources = render_settings(stage)
        if volumes and not requested_settings:
            raise ValueError("NuRec volume has no authored renderSettings in root or referenced layers")
        app.reset_render_settings()
        effective = {}
        for key, value in requested_settings.items():
            setting_path = "/" + key.replace(":", "/").lstrip("/")
            settings.set(setting_path, value)
            effective[setting_path] = settings.get(setting_path)
        receipt["render_settings_sources"] = settings_sources
        receipt["applied_render_settings"] = effective
        receipt["gpu_settings"] = {key: settings.get(key) for key in
                                   ("/renderer/activeGpu", "/renderer/multiGpu/enabled", "/physics/cudaDevice")}
        receipt["used_layers"] = [layer.identifier for layer in stage.GetUsedLayers()]
        rep.orchestrator.set_capture_on_play(False)
        product = rep.create.render_product(args.camera, (args.width, args.height), name="real2sim2296_capture")
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        targets = stage.GetPrimAtPath(product.path).GetRelationship("camera").GetTargets()
        receipt["render_product"] = {"path": product.path, "camera_targets": [str(path) for path in targets]}
        if [str(path) for path in targets] != [args.camera]:
            raise ValueError(f"Render product uses an unexpected camera: {targets}")
        snapshot("camera_and_settings_verified")

        def capture(label):
            snapshot("capturing_" + label)
            for step in range(args.frames):
                app.update()
                if step % 16 == 0:
                    progress("render_warmup", image=label, frame=step, elapsed_seconds=time.monotonic() - started)
            faulthandler.dump_traceback_later(90)
            try:
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=True)
            finally:
                faulthandler.cancel_dump_traceback_later()
            data = np.asarray(annotator.get_data()).copy()
            if data.dtype != np.uint8 or data.ndim != 3 or data.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Unexpected RGB annotator data: {data.shape} {data.dtype}")
            if data.shape[2] not in (3, 4):
                raise RuntimeError(f"Unexpected channel count: {data.shape}")
            return data[:, :, :3]

        rgb = capture("scene")
        Image.fromarray(rgb).save(output)
        receipt["output"] = {"path": str(output), "sha256": sha256(output), "shape": list(rgb.shape),
                             "min": int(rgb.min()), "max": int(rgb.max()),
                             "channel_mean": rgb.mean(axis=(0, 1)).tolist(),
                             "channel_std": rgb.std(axis=(0, 1)).tolist()}
        if args.hidden_control:
            if not volumes:
                raise ValueError("Hidden-volume control requested but scene has no NuRec volume")
            for prim in volumes:
                UsdGeom.Imageable(prim).MakeInvisible()
            control = capture("hidden_volumes")
            Image.fromarray(control).save(control_path)
            difference = np.abs(rgb.astype(np.int16) - control.astype(np.int16))
            mask = difference.max(axis=2) > 8
            receipt["hidden_control"] = {"path": str(control_path), "sha256": sha256(control_path),
                                          "max_abs_difference": int(difference.max()),
                                          "mean_abs_difference": float(difference.mean()),
                                          "pixels_different_by_more_than_8": int(mask.sum()),
                                          "different_fraction": float(mask.mean()),
                                          "gaussian_contribution_observed": bool(mask.sum() >= 16)}
        receipt["source_unchanged"] = sha256(source) == before
        if not receipt["source_unchanged"]:
            raise RuntimeError("Source stage file changed during rendering")
        receipt["status"] = "rendered"
        progress("render_complete", output=receipt["output"], hidden_control=receipt.get("hidden_control"))
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        receipt["elapsed_seconds_before_shutdown"] = time.monotonic() - started
        snapshot(receipt["status"])
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        if app is not None:
            try:
                if annotator is not None:
                    annotator.detach()
                if product is not None:
                    product.destroy()
            finally:
                app.close()


if __name__ == "__main__":
    main()
