# Textured, articulated reconstruction of IMG_2296

The user clarified on 2026-09-12 that the deliverable must consist of visually
similar textured geometric models with articulation. The opened Gaussian
reconstruction is useful reference evidence but does not satisfy that request.

## Evidence and constraints

Use the original MOV, the 68-entry source inventory, calibrated image views,
and the partial multiview geometry in `runs/real2sim-2296/20260912T0614Z`.
Preserve those artifacts. The textured mesh viewer replaces the earlier live
Gaussian viewer; both source scene files remain unchanged. Keep this implementation and its
outputs under the existing real2sim experiment and a new `physical` run.
The movie contains no established metric depth; camera reconstruction is
imperfect. Observed appearance and mechanical hypotheses must remain distinct.
Do not upload the video, replace original evidence, or claim real-world metric
validation, measured joint dynamics, or G1 transfer from rendered images.

## Outcome and implementation

Author a single OpenUSD scene of separate mesh assets, with source-derived
textures/materials, physically modeled cabinet carcasses and openings,
individual door/drawer/appliance links, joint limits and collision shapes.
Include the two distinct islands, three sink stations, perimeter cabinetry,
appliances, stools, tables, carts, architectural surfaces, and visible small
task objects. Use Blender where it improves mesh/material authoring; OpenUSD
owns final scene composition and Isaac articulation/physics.

Estimate asset placement and proportions from calibrated source views and
multiview surface evidence. Explicitly record any nominal scale, inferred
hidden geometry, unresolved objects and material/dynamic assumptions.
Preserve identifiable visual details using actual source texture crops where
appropriate. Generated generic furniture or splats alone are not acceptance.

## Observable acceptance and verification

1. The default scene renders mesh geometry with the splat layer absent.
2. The source inventory maps to concrete assets, with omissions identified.
3. Visible outlines, surface colors, textures and placement are compared in
   several matching source cameras; report disagreements, not only one view.
4. Cabinet doors, drawers and appliance doors have independent rigid links,
   valid joint frames, sensible travel limits and usable handles. Tests must
   exercise opening/closing under simulation, not only authored animation.
5. Collision geometry preserves openings and usable counter/shelf surfaces;
   support/contact tests use the same composed assets visible in the scene.
6. The scene opens and renders in Isaac; a recorded articulation demonstration
   and screenshots show physical assets. No completion claim based on schemas
   or image resemblance alone.
7. Keep a reproducible source/configuration manifest, verification receipts,
   and a report of work, failures, estimates and remaining deployment gaps.

## Unresolved evidence

Absolute dimensions, actual hinge/slide hardware and limits, mass/friction,
unseen cabinet interiors, exact G1 hand and LiDAR configuration, and real/sim
task correlation are unavailable. Use documented adjustable estimates where
needed for an operable model, without representing them as measured.
