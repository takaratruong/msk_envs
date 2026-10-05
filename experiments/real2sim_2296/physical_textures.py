#!/usr/bin/env python3
"""Rectify explicitly selected IMG_2296 training-frame texture regions.

All delivered pixels are resampled from the recorded RGB frames. There is no
inpainting, de-lighting, sharpening, synthesis, learned upscaling, or upload.
The selected quadrilateral and the rectangular target aspect are authoring
choices, not metric calibration. See the generated manifest for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def texture(key, frame, quad, size, kind, intended, notes, roughness, **extra):
    return dict(key=key, frame=f"frame_{frame:06d}.jpg", quad=quad,
                size=size, kind=kind, intended=intended, notes=notes,
                roughness=roughness, **extra)


# Quad order is the intended raster TL, TR, BR, BL, in source pixel-centre
# coordinates. These are deliberately inset from occluders and material edges.
SPECS = [
    texture("wood_maple", 1620,
            [[466,470],[840,601],[818,963],[489,788]], [384,384],
            "material_patch", "Sorting-island side and related maple laminate panels",
            "Clean central side-panel region, below the countertop shadow. "
            "Grain runs primarily along raster rows from top toward bottom. "
            "The square crop does not establish a physical texel density.", .58,
            grain_direction="vertical in the delivered raster"),
    texture("wood_maple_wall", 1009,
            [[1146,646],[1354,655],[1343,815],[1142,801]], [208,160],
            "material_patch", "Wall-cabinet maple laminate; alternative observed illumination",
            "Inset lower right door patch, below and to the right of its handle. "
            "This is a separate observation, not a color correction of wood_maple.", .58,
            grain_direction="vertical in the delivered raster"),
    texture("wood_maple_front", 1573,
            [[458,664],[624,624],[623,986],[478,1030]], [176,384],
            "material_patch", "Maple front panel of sorting-island landfill door",
            "Clean grain to the left of the landfill sign. This patch preserves the "
            "stronger grain of the front panel without its label or handle.", .58,
            grain_direction="vertical in the delivered raster"),
    texture("floor_gray_tile", 559,
            [[542,798],[855,733],[953,1022],[583,1070]], [352,288],
            "material_patch", "Light gray mottled resilient floor tile",
            "Interior floor patch, excluding the charcoal border, drain, and cabinet. "
            "This is a patch within the tiled floor, not a measured complete tile. "
            "Tile size and repeat density must be authored separately.", .63),
    texture("floor_charcoal_border", 559,
            [[1002,740],[1118,729],[1314,1008],[1153,1032]], [128,288],
            "material_patch", "Dark charcoal floor border around work areas",
            "Patch inside the dark floor strip. Recorded gloss and illumination remain baked in.", .48),
    texture("white_countertop", 649,
            [[95,130],[450,116],[526,500],[81,527]], [384,384],
            "material_patch", "White finely speckled countertop",
            "Close unobstructed region of the sorting-island counter, excluding signs and rim. "
            "The selected planar quad is rectified for texture use; its square target "
            "does not measure the real speckle scale or plane metric.", .36),
    texture("label_landfill", 1573,
            [[651,561],[881,510],[860,777],[647,851]], [240,311],
            "opaque_decal", "Original gray landfill sign on sorting-island door",
            "Original title, icons, and footer retained. Nominal portrait paper aspect "
            "8.5:11 is an authoring assumption; no text was recreated.", .67),
    texture("label_compost", 1573,
            [[1139,432],[1284,393],[1242,621],[1105,673]], [160,207],
            "opaque_decal", "Original green compost sign on sorting-island door",
            "Original title, icons, and footer retained. Nominal portrait paper aspect "
            "8.5:11 is an authoring assumption; fine print remains resolution-limited.", .67),
    texture("label_recycle", 1573,
            [[1479,348],[1580,320],[1524,522],[1428,556]], [144,186],
            "opaque_decal", "Original blue recycle sign on sorting-island door",
            "Oblique source; rectification cannot restore missing horizontal detail. "
            "Nominal portrait paper aspect 8.5:11 is an authoring assumption.", .67),
    texture("rim_compost", 649,
            [[287,856],[494,628],[532,643],[326,870]], [320,48],
            "opaque_decal", "Green compost strip on the counter rim",
            "Actual horizontal counter strip, rotated into reading orientation. "
            "Its length:height ratio is an authoring estimate.", .67),
    texture("rim_recycle", 649,
            [[793,304],[877,205],[911,217],[822,312]], [144,26],
            "opaque_decal", "Blue recycle strip on the counter rim",
            "Actual horizontal counter strip, rotated into reading orientation. "
            "Its length:height ratio is an authoring estimate.", .67),
    texture("rim_landfill", 1573,
            [[676,257],[877,231],[893,240],[690,268]], [208,32],
            "opaque_decal", "Gray landfill strip on the counter rim",
            "Strongly foreshortened actual strip. Vertical detail is limited to "
            "approximately ten source pixels; the rectified height adds no new detail.", .67),
    texture("fridge_notice", 885,
            [[1617,517],[1744,520],[1737,651],[1611,648]], [128,136],
            "opaque_decal", "Original orange taped notice on refrigerator door",
            "Preserves the notice and tape as recorded, including small exposed metal "
            "areas and their reflection. Text is not resolved or retyped.", .67),
    texture("display_kitchen", 937,
            [[753,114],[1074,139],[1054,326],[745,317]], [320,180],
            "display_snapshot", "Kitchen-wall display near dishwasher and wall sink",
            "One recorded instant. Black image margins are retained; outer bezel excluded. "
            "16:9 target aspect is an authoring assumption. Display animation is not recovered.", .25),
    texture("display_dual_left", 858,
            [[1154,68],[1608,58],[1584,317],[1148,308]], [448,252],
            "display_snapshot", "Left of the two displays above the toaster-side wall sink",
            "One recorded instant, independently timed from display_dual_right. "
            "16:9 target aspect is an authoring assumption. Outer bezel excluded.", .25),
    texture("display_dual_right", 885,
            [[537,121],[1048,167],[1030,441],[533,428]], [512,288],
            "display_snapshot", "Right of the two displays above the toaster-side wall sink",
            "One recorded instant, independently timed from display_dual_left. "
            "16:9 target aspect is an authoring assumption. Outer bezel excluded.", .25),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def font(size=18):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def make_contacts(out, records):
    for sheet_name, selected in [
        ("materials_contact.png", [x for x in records if x["kind"] == "material_patch"]),
        ("decals_contact.png", [x for x in records if x["kind"] == "opaque_decal"]),
        ("displays_contact.png", [x for x in records if x["kind"] == "display_snapshot"]),
    ]:
        cols, cell_w, cell_h = 3, 512, 480
        canvas = Image.new("RGB", (cols*cell_w, ((len(selected)+cols-1)//cols)*cell_h), "#252932")
        draw = ImageDraw.Draw(canvas)
        for i, rec in enumerate(selected):
            x, y = i % cols * cell_w, i // cols * cell_h
            im = Image.open(out / rec["path"]).convert("RGB")
            # Preview scaling only. Source-derived delivered PNG is unaffected.
            im.thumbnail((cell_w-28, cell_h-94), Image.Resampling.LANCZOS)
            canvas.paste(im, (x+(cell_w-im.width)//2, y+12+(cell_h-94-im.height)//2))
            draw.text((x+14,y+cell_h-76), rec["key"], fill="white", font=font(21))
            draw.text((x+14,y+cell_h-47), f"{rec['width']} x {rec['height']} px | {rec['source']['time_seconds']:.3f}s", fill="#b8c4d6", font=font(17))
            draw.text((x+14,y+cell_h-23), rec["source"]["frame"], fill="#b8c4d6", font=font(15))
        canvas.save(out / sheet_name)


def write_usage(out, records):
    table = "\n".join(f"| `{r['key']}` | [{r['path']}]({r['path']}) | {r['width']} x {r['height']} | {r['source']['frame']} ({r['source']['time_seconds']:.3f}s) |" for r in records)
    (out / "README.md").write_text(f"""# Source-derived physical-scene textures

All texture PNG pixels are perspective-resampled from the recorded 1920 x 1080
RGB training frames. The manifest binds each source image, source video,
frame manifest, quad, homography, script, and delivered PNG by SHA-256.
These textures preserve photographed illumination, reflections, camera color,
compression, and blur. They are source appearance proxies, **not measured albedo**.
There is no generative fill, texture synthesis, sharpening, exposure correction,
text recreation, or uploaded content.

| Texture key | File | Raster pixels | Source |
|---|---|---:|---|
{table}

Use PNGs as sRGB inputs and let the renderer decode sRGB to linear. Do not load
them as raw linear data and do not apply another gamma conversion manually.
For material patches, use the RGB as a base-color appearance proxy; lighting
must account for the baked shading. Roughness and metallic entries in the
manifest are adjustable scalar guesses, not measured maps. No normals,
displacement, roughness, metalness, opacity, or metric texel density were inferred
from RGB. White-countertop speckles and floor markings are appearance only.

The standard files use normal top-left raster storage: x increases right,
y increases down. They have no mirrored or vertical flip baked in. For a
front-facing quad whose vertices are TL, TR, BR, BL, conventional bottom-left
UVs are (0,1), (1,1), (1,0), (0,0); a top-left UV API uses (0,0), (1,0),
(1,1), (0,1). Verify the word 'landfill' is upright in the selected renderer
and perform any API-required flip once. Wood grain runs vertically in both
wood maps. Use a complete label once across its front face; labels are opaque
rectangles and include their recorded background. They are not alpha cutouts.

Material patches are not seamless and have no measured repeat spacing. A
single UV span per visible panel preserves the supplied grain direction.
If repeating a patch, set physical scale from the scene's explicit dimension
assumptions and inspect repetition and seams. No mirrored repeat or synthesized
continuation is delivered. The light-floor patch is not a complete measured tile.

Display files are individually time-stamped static appearances. They can feed
an emissive color input with an explicitly estimated strength; no calibrated
display radiance, emission spectrum, or animation is available. The two display
snapshots come from different instants. The fridge notice remains unreadable in
places; the recorded image was preserved without recreating its text.

Rectification chooses a planar quad and a useful rectangular aspect, without
claiming physical calibration. The manifest includes actual source edge lengths,
source area, and local resampling scale; output pixel counts do not establish
recovered detail. Source JPEG dimensions are the resolution ceiling.

The contact sheets and per-texture source-context images are inspection media,
not additional source texture content. `inspection_receipt.json`, when present,
records which exact output hashes were visually inspected after generation.
The `inspection/` directory contains exploratory source-selection media; the
manifest's quads and the final source-context files are authoritative.
""")
    cards = []
    for rec in records:
        esc = html.escape
        cards.append(f"""<article><h2>{esc(rec['key'])}</h2><div class="pair"><a href="{esc(rec['path'])}"><img src="{esc(rec['path'])}" alt="Rectified {esc(rec['key'])}"></a><a href="{esc(rec['context_path'])}"><img src="{esc(rec['context_path'])}" alt="Source with selected quad"></a></div><p>{esc(rec['intended_material'])}</p><p>{rec['width']} x {rec['height']} pixels; source {esc(rec['source']['frame'])}, {rec['source']['time_seconds']:.6f}s.</p><p>{esc(rec['notes'])}</p><p class="hash">PNG SHA-256 {esc(rec['sha256'])}</p></article>""")
    (out / "index.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>IMG_2296 source-derived textures</title><style>body{font:16px/1.5 system-ui,sans-serif;background:#171b22;color:#e6edf5;max-width:1280px;margin:32px auto;padding:0 20px}a{color:#82bdff}article{padding:20px;border:1px solid #3d4757;margin:24px 0;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1.5fr;gap:24px;align-items:center}.pair img{max-width:100%;max-height:430px;object-fit:contain}.hash{font:12px monospace;overflow-wrap:anywhere}h2{font-size:22px}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style><h1>Source-derived physical-scene textures</h1><p>Recorded RGB patches and labels, perspective-rectified with preserved provenance. These are photographed appearance with baked lighting, not measured albedo. Material properties and texture scale are authoring assumptions.</p><p><a href="manifest.json">Manifest</a> · <a href="README.md">Material and UV usage</a> · <a href="inspection_receipt.json">Visual inspection receipt (if completed)</a></p>""" + "".join(cards) + "</html>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Check existing manifest bindings without changing files")
    args = parser.parse_args()
    out, run = args.output.resolve(), args.source_run.resolve()
    if args.verify_only:
        m = json.loads((out / "manifest.json").read_text())
        checks = {str(p): sha256(p) == h for p, h in m["input_files_sha256"].items()}
        checks.update({k: sha256(out/k) == h for k, h in m["artifacts_sha256"].items()})
        assert all(checks.values()), [k for k, ok in checks.items() if not ok]
        print(json.dumps({"verified_hash_bindings": len(checks), "all_match": True}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_context").mkdir(exist_ok=True)
    frames_path = run / "frames.json"
    frames_manifest = json.loads(frames_path.read_text())
    frames = {x["name"]: x for x in frames_manifest["frames"]}
    source_movie = Path(frames_manifest["source_path"])
    assert sha256(source_movie) == frames_manifest["source_sha256"], "Source movie hash mismatch"
    inputs = {str(source_movie): frames_manifest["source_sha256"],
              str(frames_path): sha256(frames_path), str(Path(__file__).resolve()): sha256(__file__)}
    inventory = run / "semantic_inventory" / "objects.json"
    if inventory.exists():
        inputs[str(inventory)] = sha256(inventory)
    records = []
    for spec in SPECS:
        frame = frames[spec["frame"]]
        assert frame["split"] == "train", f"Held-out source disallowed: {spec['frame']}"
        source = Path(frame["image_path"])
        assert sha256(source) == frame["sha256"], f"Source image hash mismatch: {source}"
        inputs[str(source)] = frame["sha256"]
        src = cv2.imread(str(source), cv2.IMREAD_COLOR)
        assert src is not None and tuple(src.shape[:2]) == (frame["height"], frame["width"])
        quad = np.asarray(spec["quad"], dtype=np.float32)
        assert quad.shape == (4,2) and np.isfinite(quad).all()
        assert (quad >= 0).all() and (quad[:,0] < src.shape[1]).all() and (quad[:,1] < src.shape[0]).all()
        assert cv2.isContourConvex(quad) and cv2.contourArea(quad, oriented=True) > 0
        width, height = spec["size"]
        dst = np.asarray([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(quad, dst)
        im = cv2.warpPerspective(src, transform, (width,height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255,0,255))
        # Every destination sample must map inside the source. No extrapolated borders.
        yy, xx = np.mgrid[0:height,0:width]
        q = cv2.perspectiveTransform(np.stack([xx,yy],axis=-1).astype(np.float64).reshape(-1,1,2), np.linalg.inv(transform)).reshape(height,width,2)
        assert q.min() >= 3 and q[:,:,0].max() < src.shape[1]-3 and q[:,:,1].max() < src.shape[0]-3
        edges = np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1)
        # Singular values quantify the local sampling footprint in input pixels per
        # output pixel, including perspective; values below one mean interpolation.
        jac = np.stack([np.gradient(q, axis=1), np.gradient(q, axis=0)],axis=-1)
        scales = np.linalg.svd(jac.reshape(-1,2,2), compute_uv=False)
        file_path = f"{spec['key']}.png"
        assert cv2.imwrite(str(out/file_path), im)
        context_path = f"source_context/{spec['key']}.jpg"
        context = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(context)
        points = [tuple(x) for x in spec["quad"]]
        draw.line(points+[points[0]], fill=(255,56,120), width=5)
        for i, point in enumerate(points):
            x,y = point
            draw.ellipse((x-5,y-5,x+5,y+5), fill=(255,255,0))
            draw.text((x+7,y-23), ["TL","TR","BR","BL"][i], font=font(22), fill="yellow", stroke_width=2, stroke_fill="black")
        draw.rectangle((0,0,src.shape[1],43), fill=(20,25,34))
        draw.text((12,8), f"{spec['key']} | {spec['frame']} | PTS {frame['time_seconds']:.6f}s | source {src.shape[1]}x{src.shape[0]}", font=font(21), fill="white")
        context.save(out/context_path, quality=94)
        record = dict(key=spec["key"], path=file_path, sha256=sha256(out/file_path),
                      context_path=context_path, context_sha256=sha256(out/context_path),
                      width=width, height=height, channels=3, color_space="sRGB", kind=spec["kind"],
                      intended_material=spec["intended"], notes=spec["notes"],
                      source=dict(frame=spec["frame"], image_path=str(source), sha256=frame["sha256"],
                                  source_index=frame["source_index"], time_seconds=frame["time_seconds"],
                                  split=frame["split"], width=frame["width"], height=frame["height"]),
                      source_quad_tl_tr_br_bl_pixel_centres=spec["quad"],
                      homography_source_to_output=transform.tolist(),
                      available_source_resolution=dict(edge_lengths_top_right_bottom_left_px=edges.tolist(),
                          polygon_area_source_pixels=float(cv2.contourArea(quad)),
                          pixel_footprint_singular_values_source_px_per_output_px=dict(
                              min=float(scales.min()), p05=float(np.quantile(scales,.05)),
                              median=float(np.median(scales)), p95=float(np.quantile(scales,.95)), max=float(scales.max())),
                          source_jpeg_resolution_ceiling=[src.shape[1],src.shape[0]],
                          no_new_detail_from_interpolation=True),
                      process=dict(method="cv2.getPerspectiveTransform + cv2.warpPerspective", interpolation="INTER_CUBIC",
                                   source_image_border_extrapolation=False, color_adjustment=False,
                                   generative_content=False, measured_aspect_ratio=False),
                      material_assumptions=dict(measured_albedo=False, contains_baked_illumination=True,
                          source_camera_color_and_blur_retained=True, roughness=spec["roughness"],
                          metallic=0.0, scalar_properties_measured=False, opacity=1.0,
                          suggested_input="emissiveColor" if spec["kind"] == "display_snapshot" else "baseColor",
                          emission_strength_measured=False if spec["kind"] == "display_snapshot" else None),
                      uv=dict(raster_origin="top_left", raster_x="right", raster_y="down",
                              front_quad_order=["TL","TR","BR","BL"],
                              conventional_bottom_left_st=[[0,1],[1,1],[1,0],[0,0]],
                              top_left_uv=[[0,0],[1,0],[1,1],[0,1]],
                              baked_flip=False, seamless=False, physical_texel_density_measured=False,
                              grain_direction=spec.get("grain_direction")))
        records.append(record)
    make_contacts(out, records)
    write_usage(out, records)
    paths = [x["path"] for x in records] + [x["context_path"] for x in records]
    paths += ["materials_contact.png", "decals_contact.png", "displays_contact.png", "README.md", "index.html"]
    manifest = dict(schema="source-derived-physical-textures/v1", created_at=datetime.now(timezone.utc).isoformat(),
                    source_run=str(run), script=str(Path(__file__).resolve()), input_files_sha256=inputs,
                    software=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__, platform=platform.platform()),
                    texture_count=len(records), source_splits=["train"], textures=records,
                    all_output_pixels_source_derived=True, measured_albedo=False, metric_texture_scale_measured=False,
                    visual_inspection_status="Requires a separately hash-bound inspection_receipt.json after viewing outputs",
                    artifacts_sha256={p:sha256(out/p) for p in paths})
    save_json(out/"manifest.json", manifest)
    print(json.dumps({"manifest":str(out/"manifest.json"), "manifest_sha256":sha256(out/"manifest.json"),
                      "texture_count":len(records), "output":str(out)}, indent=2))


if __name__ == "__main__":
    main()
