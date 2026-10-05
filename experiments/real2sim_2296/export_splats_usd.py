#!/usr/bin/env python3
"""Convert standard 3DGS PLY to legacy NuRec USDZ without changing coordinates.

CPU-only bridge to two verified functions from NVIDIA 3DGRUT's NuRec template.
Dependencies: numpy, plyfile, msgpack, usd-core. No Isaac or CUDA initialization.
See USD_EXPORT.md for representation, precision and renderer limitations.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Dict
import urllib.request
import zipfile

import msgpack
import numpy as np
from plyfile import PlyData
from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol


UPSTREAM_COMMIT = "a37ef721012dea0f29c0fcfff2d525023b4e854a"
TEMPLATE_PATH = "threedgrut/export/usd/nurec/templates.py"
TEMPLATE_SHA256 = "8632ad7c712ce3fdc59263ee725ef838869f2b972f00c124cdb8ad576599d57f"
TEMPLATE_URL = f"https://raw.githubusercontent.com/nv-tlabs/3dgrut/{UPSTREAM_COMMIT}/{TEMPLATE_PATH}"
SOURCE_PREFIX = ".gaussians_nodes.gaussians."
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_template(cache: Path, offline: bool):
    """Retain upstream source/license header and execute only its two functions."""
    path = cache / f"3dgrut-{UPSTREAM_COMMIT}-nurec-templates.py"
    if path.exists():
        source = path.read_bytes()
    elif offline:
        raise ValueError(f"Pinned upstream template is missing in offline mode: {path}")
    else:
        with urllib.request.urlopen(TEMPLATE_URL, timeout=45) as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != TEMPLATE_SHA256:
            raise ValueError("Downloaded upstream template failed its pinned SHA256 check")
        cache.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing another run's cached source.
        try:
            with path.open("xb") as stream:
                stream.write(source)
        except FileExistsError:
            source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != TEMPLATE_SHA256:
        raise ValueError(f"Cached upstream template failed its pinned SHA256 check: {path}")
    parsed = ast.parse(source, filename=str(path))
    wanted = {"_fill_state_dict_tensors", "fill_3dgut_template"}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise ValueError("Pinned template does not contain the expected functions")
    # The upstream module imports an unused NamedSerialized from the full
    # training tree. Selecting only these function ASTs avoids that dependency;
    # their bodies/defaults are unchanged and the complete source is retained.
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np, "Any": Any, "Dict": Dict}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["fill_3dgut_template"], path


def load_ply(path: Path):
    data = PlyData.read(str(path))
    try:
        vertices = data["vertex"]
    except KeyError as error:
        raise ValueError("PLY is missing its vertex element") from error
    names = set(vertices.data.dtype.names or [])
    required = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    required += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    missing = sorted(set(required) - names)
    if missing:
        raise ValueError(f"Expected standard Gaussian PLY; missing properties: {missing}")
    for prefix, count in (("scale_", 3), ("rot_", 4), ("f_dc_", 3)):
        if {n for n in names if n.startswith(prefix)} != {f"{prefix}{i}" for i in range(count)}:
            raise ValueError(f"Expected exactly {count} {prefix} properties")
    rest = [name for name in names if name.startswith("f_rest_")]
    if set(rest) != {f"f_rest_{i}" for i in range(len(rest))}:
        raise ValueError("Higher-order SH properties must be consecutively numbered from f_rest_0")
    degrees = {0: 0, 9: 1, 24: 2, 45: 3}
    if len(rest) not in degrees:
        raise ValueError(f"Only complete SH degrees 0–3 are supported, got {len(rest)} f_rest properties")
    for name in required + rest:
        dtype = vertices.data.dtype.fields[name][0]
        if dtype.kind != "f" or dtype.itemsize != 4:
            raise ValueError(f"Expected scalar float32 property {name}; got {dtype}. Preserve the source precision explicitly before export.")
    degree = degrees[len(rest)]
    count = len(vertices)
    if not count:
        raise ValueError("Empty Gaussian PLY")

    def columns(keys):
        return np.stack([vertices[key] for key in keys], axis=1).astype(np.float32)

    attributes = {
        "positions": columns(["x", "y", "z"]),
        "rotations": columns([f"rot_{i}" for i in range(4)]),
        "scales": columns([f"scale_{i}" for i in range(3)]),
        "densities": columns(["opacity"]),
        "features_albedo": columns([f"f_dc_{i}" for i in range(3)]),
    }
    # Standard 3DGS PLY is channel-major; NuRec expects coefficient-major RGB.
    if rest:
        attributes["features_specular"] = columns([f"f_rest_{i}" for i in range(len(rest))]).reshape(
            count, 3, -1).transpose(0, 2, 1).reshape(count, -1)
    else:
        attributes["features_specular"] = np.empty((count, 0), dtype=np.float32)
    for name, values in attributes.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {name}")
        if values.size and float(np.max(np.abs(values))) > float(np.finfo(np.float16).max):
            raise ValueError(f"{name} exceeds finite float16 range; refusing lossy overflow")
    rotations = attributes["rotations"].astype(np.float16).astype(np.float32)
    if np.any(np.linalg.norm(rotations, axis=1) < 1e-12):
        raise ValueError("A quaternion is zero or becomes zero after float16 conversion")
    with np.errstate(over="ignore", under="ignore"):
        activated_scales = np.exp(attributes["scales"].astype(np.float16).astype(np.float32))
    if not np.isfinite(activated_scales).all() or np.any(activated_scales <= 0):
        raise ValueError("Log-scales yield invalid float32 scales after quantization")
    return attributes, degree


def quantization_report(attributes):
    positions = attributes["positions"]
    quantized = positions.astype(np.float16).astype(np.float32)
    differences = quantized.astype(np.float64) - positions.astype(np.float64)
    distances = np.linalg.norm(differences, axis=1)
    return {
        "units": "raw_input_units; metric_scale_unverified",
        "max_xyz_component_abs_error": float(np.abs(differences).max()),
        "max_point_l2_error": float(distances.max()),
        "p95_point_l2_error": float(np.percentile(distances, 95)),
        "rms_point_l2_error": float(np.sqrt(np.mean(distances ** 2))),
        "input_xyz_min": positions.min(axis=0).tolist(),
        "input_xyz_max": positions.max(axis=0).tolist(),
        "export_xyz_min": quantized.min(axis=0).tolist(),
        "export_xyz_max": quantized.max(axis=0).tolist(),
        "max_abs_error_by_tensor": {
            key: float(np.max(np.abs(value.astype(np.float16).astype(np.float64) - value))) if value.size else 0.0
            for key, value in attributes.items()
        },
    }


def create_stage(positions, provenance):
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("upAxis", "Z")
    # An encoding convention only: the caller must establish metric scale in
    # its common parent transform. No position or radius is scaled here.
    stage.SetMetadata("metersPerUnit", 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.SetMetadataByDictKey("customLayerData", "real2sim", {
        "sourcePlySha256": provenance["source_ply"]["sha256"],
        "sourceUnits": "raw_reconstruction_units",
        "metricScaleVerified": False,
        "renderVerified": False,
        "upstreamCommit": UPSTREAM_COMMIT,
    })
    # Match the pinned legacy exporter, not the newer ParticleField renderer.
    settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", settings)
    volume = UsdVol.Volume.Define(stage, "/World/gauss")
    prim = volume.GetPrim()
    volume.AddTransformOp().Set(Gf.Matrix4d(1.0))
    prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)
    quantized = positions.astype(np.float16).astype(np.float32)
    low = np.minimum(positions.min(axis=0), quantized.min(axis=0))
    high = np.maximum(positions.max(axis=0), quantized.max(axis=0))
    low_vec, high_vec = Gf.Vec3f(*map(float, low)), Gf.Vec3f(*map(float, high))
    prim.GetAttribute("extent").Set([low_vec, high_vec])
    prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0))
    prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(low_vec)
    prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(high_vec)
    prim.CreateRelationship("proxy")
    for child, role, datatype in (("density_field", "density", "float"),
                                   ("emissive_color_field", "emissiveColor", "float3")):
        field = stage.DefinePrim(f"/World/gauss/{child}", "OmniNuRecFieldAsset")
        volume.CreateFieldRelationship(role, field.GetPath())
        field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set("./model.nurec")
        for name, value in (("fieldName", role), ("fieldDataType", datatype), ("fieldRole", role)):
            field.CreateAttribute(name, Sdf.ValueTypeNames.Token).Set(value)
        if role == "emissiveColor":
            for channel, vector in zip("RGB", ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))):
                field.CreateAttribute(f"omni:nurec:ccm{channel}", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*vector))
    return stage


def write_aligned_entry(archive, name, content):
    """ZIP_STORED USDZ members must start on a 64-byte boundary."""
    offset = archive.fp.tell() + 30 + len(name.encode("utf-8"))
    padding = (-offset) % 64
    if 0 < padding < 4:
        padding += 64
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    if padding:
        info.extra = struct.pack("<HH", 0x1986, padding - 4) + b"\0" * (padding - 4)
    archive.writestr(info, content)


def verify_package(path, attributes):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist()[0] != "default.usda" or archive.testzip() is not None:
            raise ValueError("Invalid USDZ ordering or CRC")
        for info in archive.infolist():
            with path.open("rb") as stream:
                stream.seek(info.header_offset)
                header = stream.read(30)
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            if (info.header_offset + 30 + name_length + extra_length) % 64:
                raise ValueError("Unaligned USDZ member")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("USDZ member has unsupported ZIP compression")
        payload = msgpack.unpackb(gzip.decompress(archive.read("model.nurec")), raw=False)
        state = payload["nre_data"]["state_dict"]
        for key, expected in attributes.items():
            stored_name = SOURCE_PREFIX + key
            actual = np.frombuffer(state[stored_name], dtype=np.float16).reshape(state[stored_name + ".shape"])
            if not np.array_equal(actual, expected.astype(np.float16)):
                raise ValueError(f"NuRec tensor round-trip failed: {key}")
    stage = Usd.Stage.Open(str(path))
    if not stage or not stage.GetPrimAtPath("/World/gauss"):
        raise ValueError("OpenUSD could not read packaged Volume")
    volume = UsdGeom.Xformable(stage.GetPrimAtPath("/World/gauss"))
    if not np.array_equal(np.asarray(volume.ComputeLocalToWorldTransform(Usd.TimeCode.Default())), IDENTITY):
        raise ValueError("Unexpected non-identity export transform")
    for child in ("density_field", "emissive_color_field"):
        value = stage.GetPrimAtPath(f"/World/gauss/{child}").GetAttribute("filePath").Get()
        if not value or not value.resolvedPath:
            raise ValueError(f"Unresolved packaged NuRec payload: {child}")
    return {"usd_open": "passed", "identity_transform": "passed", "payload_resolution": "passed",
            "tensor_round_trip": "passed", "usdz_alignment_crc": "passed",
            "isaac_render": "not_run", "collision_geometry": "not_included"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Standard 3DGS PLY with raw parameters")
    parser.add_argument("--output", type=Path, required=True, help="New legacy NuRec .usdz")
    parser.add_argument("--upstream-cache", type=Path, help="Default: output parent/export_upstream")
    parser.add_argument("--offline", action="store_true", help="Require the verified upstream template in cache")
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    if output.suffix.lower() != ".usdz":
        raise ValueError("--output must end in .usdz")
    if output.exists() or receipt.exists():
        raise ValueError("Output or receipt already exists; choose a new output name")
    before = sha256(source)
    attributes, degree = load_ply(source)
    if sha256(source) != before:
        raise ValueError("Input PLY changed during conversion")
    fill_template, template_path = load_template(args.upstream_cache or output.parent / "export_upstream", args.offline)
    report = quantization_report(attributes)
    provenance = {
        "schema": "real2sim-nurec-export/v1",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "source_ply": {"path": os.path.relpath(source, output.parent), "sha256": before, "authority": True},
        "converter": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "upstream": {"commit": UPSTREAM_COMMIT, "template_url": TEMPLATE_URL,
                     "template_sha256": TEMPLATE_SHA256, "cache_path": str(template_path.resolve()),
                     "license": "Apache-2.0", "bridge": "unchanged ASTs of two pinned template functions"},
        "gaussian_count": len(attributes["positions"]), "sh_degree": degree,
        "format": "legacy NuRec Volume with float16 pre-activation tensors",
        "coordinate_transform": IDENTITY, "normalization_applied": False,
        "metric_scale_verified": False, "up_axis": "Z metadata only; coordinates unchanged",
        "quantization": report,
        "runtime_packages": {name: importlib.metadata.version(name) for name in ["numpy", "plyfile", "msgpack", "usd-core"]},
        "render_status": "not_run; installed Isaac capability requires a separate render check",
    }
    # Defaults match _get_default_nurec_conf in the pinned official exporter.
    template = fill_template(**attributes, n_active_features=degree,
                             density_kernel_density_clamping=True, transmittance_threshold=0.0001,
                             global_z_order=True)
    payload = gzip.compress(msgpack.packb(template), compresslevel=0, mtime=0)
    stage = create_stage(attributes["positions"], provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".nurec-export-", suffix=".usdz", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
            write_aligned_entry(archive, "default.usda", stage.GetRootLayer().ExportToString().encode())
            write_aligned_entry(archive, "model.nurec", payload)
            write_aligned_entry(archive, "PROVENANCE.json", (json.dumps(provenance, indent=2) + "\n").encode())
        checks = verify_package(temporary, attributes)
        # Exclusive hard-link publication keeps an existing artifact untouched.
        os.link(temporary, output)
        provenance["verification"] = checks
        provenance["output"] = {"path": output.name, "sha256": sha256(output), "bytes": output.stat().st_size}
        with receipt.open("x") as stream:
            json.dump(provenance, stream, indent=2)
            stream.write("\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "receipt": str(receipt), "gaussians": len(attributes["positions"]),
                      "max_xyz_error_raw_units": report["max_xyz_component_abs_error"],
                      "max_l2_error_raw_units": report["max_point_l2_error"], "isaac_render": "not_run"}, indent=2))


if __name__ == "__main__":
    main()
