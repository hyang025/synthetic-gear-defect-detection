# Lisa: two-belt motor, gear shaft and encoder assembly

Motor 20T → lower shaft 80T gives quarter-speed at the gear. Upper shaft 80T → encoder 20T returns the encoder to motor speed. Both belts are closed-loop 250 × 6 mm GT2 / 2GT, 2 mm pitch. Both device slides have 15 mm adjustment. The user reports the components are on hand; actual specifications and fit need checking.

| Folder | Current printed pieces | Contents |
| --- | ---: | --- |
| [motor holder](motor%20holder/) | 2 | Motor holder and slide base |
| [encoder holder](encoder%20holder/) | 2 | Encoder holder and slide base |
| [shaft frame](shaft%20frame/) | 2 | Integrated body and removable upper bridge |
| [gear adapter](gear%20adapter/) | 2 | Keyed sleeve and clamp cap |
| [assembly](assembly/) | 0 | Assembly CAD, shared renders, inputs and bundles |

- [Assembly manual](manual.pdf)
- [All eight STLs](assembly/output/model/lisa_two_belt_r06_stls.zip)
- [Complete CAD, STEP, drawings and sources](assembly/output/model/lisa_two_belt_r06.zip)
- [Hardware quantities](scripts/two_belt_parts.csv) and [print estimates](scripts/two_belt_print_estimates.csv)

The motor recess includes the 1.55 mm near-side relief based on the user's 7.7 mm pad diameter and 2 mm pad-to-case gap. The nominal clearance is 0.6 mm; the remaining lip is 3.85 mm. The frame uses one body with twelve gussets and a removable upper bridge. Old first-level screw flanges are removed; the narrow upper bridge seats remain. Detailed dimensions and sources are in the component notes and manual.

## Printing

A1 mini / 0.4 mm nozzle: PLA, 0.20 mm layers, six walls (approximately 2.5–2.7 mm), six top/bottom layers (1.2 mm), 40% gyroid. Both adapter pieces use 100% infill. Optional 3 mm brim. Fit, strength and fastening take priority; use supports under the integral lower bearing deck and where needed at washer lands, horizontal holes and nut-pocket roofs.
Complete set estimate: **385.7 g PLA / USD 7.71** at USD20/kg, density 1.24 g/cm³ and 5% waste. Geometric shell/infill estimate, approximately ±20%; supports, hardware, electricity and labour are excluded.

Current CAD and mesh checks passed, including two-piece frame insertion and tool access. Seven other component models are unchanged. Actual gear fit, motor fit, pulley grip and loaded stiffness need physical checks.

Rebuild from repository root with `projects/lisa/scripts/build_two_belt.sh`. Parameters, scripts and reports stay in [scripts](scripts/). Previous designs and notes are in the [Lisa archive](../../archive/lisa/). The [cleanup manifest](scripts/folder_cleanup_manifest.json) records file moves and hashes.
