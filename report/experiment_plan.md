# Healthy-memory detector on the 2D angular distribution

Plan and results, 2026-10-30. The main question is unchanged: whether and how much synthetic data helps detect 0.10–0.20 mm gear-tooth damage. How much synthetic data helps is decided after real physical data is collected; the simulation results below report the decided setup.

## Assumptions

1. ~~Most light reflects back into one central peak.~~ Dropped: in the healthy revolution the bright spot jumps to about 37° below the lens axis for about a quarter of each tooth (flank phase).
2. Damage scatters some light into reflection angles a healthy tooth does not reach.
3. That scattering does not shift the peak much.

Working replacement for 1: healthy reflections follow a fixed pattern that repeats on every tooth.

## Vector per frame

1. Screen image (512 × 512 over 150 mm) → angle of each pixel through the lens mapping, angle = arctan(position / 50 mm).
2. Fixed window of ±40° around the lens axis in 2° bins → 40 × 40 = 1,600 numbers. No centering on the peak.
3. Divide by total light.
4. Square root.

## Detector: healthy memory only

- PCA fitted on healthy revolution A; KNN distance to healthy revolution A frames in PCA space is the score.
- A tooth is flagged when its score passes the cutoff.
- PCA leftover was tried as a second score and did not separate damaged from healthy frames.
- A version with synthetic damaged frames added to the memory stays optional for the final report.

## Data roles (0.1875° spacing, `mitsuba/outputs/sequence_0p1875_25mm`)

| Frames | Role |
|---|---|
| Healthy revolution A (1,920) | KNN memory and PCA fit |
| Healthy revolution B, teeth 0–17 | Validation healthy teeth |
| Healthy revolution B, teeth 18–34 | Test healthy teeth |
| Training gears (6 random, gears 2–3; 40 cavities) | Validation cavities |
| Test gears 1, 4, 5 at Ø0.10 and Ø0.20 mm (30 cavities) | Test, used once |

The detector learns only from healthy revolution A, so the training-gear cavities and B teeth 0–17 serve as the validation set for choosing settings and the cutoff. The 5-cavity gear left over from the PatchCore experiments is not used.

Damaged label: frames within ±0.4° of a cavity crossing. False alarms are counted on healthy revolution B teeth only.

## Required metrics: FN and FP

False negatives (missed cavities) and false positives (healthy teeth flagged) are critical metrics. Every result must report:

- TP, FN, FP, and TN counts together, at the decision level: cavities for TP/FN, healthy teeth for FP/TN.
- The list of missed cavities (with size and type) and the list of flagged healthy teeth.
- The cutoff used, stated in plain words.
- Everything in one table per result. For the report, TP, FN, FP, and TN are enough (user, 2026-10-30); accuracy, detection rate, false alarms per tooth, precision, and F1 are available in the result files when needed. Accuracy depends on the cavity-to-healthy mix.

Frame-level rates and ROC area are supporting numbers only. A small per-frame false-alarm rate can still flag many teeth, because a tooth spans about 55 frames.

## Validation results (validation data only)

- First sweep, one knob at a time (ROC area): square root beat no compression, gamma 0.25, and log. PCA 5–20 beat 50–100. K 1–5 were similar; K 10 failed.
- Missed cavities are noise-limited: their change from the healthy frame at the same angle is about one-fifth of the render noise. Difference maps (`outputs/knn_pca_25mm/images/difference_maps.png`) of the missed cavities look like render noise; a clearly caught Ø0.10 mm edge cavity shows a dip at the centre and gains in the outer angles, matching assumption 2.
- Refinements to address FN and FP, with the cutoff chosen by F1:
  - 2 or 3 frames in a row: 3 in a row missed too many small cavities.
  - Averaging the vectors of 2 neighbouring frames before KNN worked best (TP 39, FN 1, FP 0, TN 18, F1 0.987). Averaging scores after KNN helped little.
  - Bin size 2°, 1°, 0.5° (finer, the user's suggestion): the best setting gave the same counts at every bin size, so 2° is kept.
- These settings were picked on the validation data, so the validation numbers are optimistic. Output files keep the earlier `tuning_` prefix (`outputs/knn_pca_25mm/tuning_*.json`).

## Decided test configuration (user, 2026-10-30)

Square root, 2° bins, PCA 5, K 3, vectors averaged over 2 neighbouring frames, a tooth flagged when a single averaged frame passes the cutoff, cutoff chosen by F1 on the validation data, saved as a fixed number, and applied unchanged to the test gears.

## Test result (run once, 2026-10-30)

Frozen cutoff 0.000947 (validation: TP 39, FN 1, FP 0, TN 18). Test gears 1, 4, 5 (30 cavities) against healthy B teeth 18–34 (17 teeth):

| TP | FN | FP | TN | Accuracy | Detection | False alarms per tooth | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| 26 | 4 | 3 | 14 | 85.1% | 86.7% | 17.6% | 89.7% | 0.881 |

- Caught: Ø0.20 mm 15/15, Ø0.10 mm 11/15; crown 6/6, edge 16/18, flank 4/6.
- Missed (all Ø0.10 mm): G1_D2 edge, G1_D5 flank, G5_D1 edge, G5_D5 flank (score 0.70–0.92 of cutoff).
- Flagged healthy segments: 18, 31, 34. Segments are cut by angle; whether the edge segments 18 and 34 reflect a counting artifact is unchecked.
- PatchCore on the same test gears found 6/15 (Ø0.10 mm) and 14/15 (Ø0.20 mm). Counted per tooth on the same segments 18–34 of its own healthy test render (about 11 frames per tooth, threshold 20.5023), it flagged 14/17 healthy teeth [Wilson 59–94%]; per frame it was 39/440.
- Files: `outputs/knn_pca_25mm/test_results.json`, `test_cutoff.json`.

## Class-survey matrix entry

Fields from the Capstone Project Description Google Doc (linked in the syllabus; not stored in the repo):

| Field | Entry |
|---|---|
| Domain and decision | Gear-tooth impact damage; whether a tooth goes to technician review |
| AI role | Flags teeth whose reflection differs from healthy teeth |
| Current process or baseline | Manual inspection with a magnifying glass and rod (user) |
| Domain requirement and source | Detection judged by probability of detection against defect size, including a90/95, the size detected with 90% probability at 95% confidence (ASTM E2862-23). The standard leaves acceptable false-call rates to the engineer or customer. |
| Evidence used | Mitsuba 3 simulation, 40 validation and 30 test cavities |
| Evaluation boundary or difficult condition | Ø0.10 mm edge and flank cavities; speed change; simulation-to-real gap |
| Strongest supported conclusion | Simulation: Ø0.20 mm 15/15, Ø0.10 mm 11/15, 3/17 healthy teeth flagged |
| Response when unsupported or after change | Re-record healthy memory, recheck the cutoff, inspect uncertain teeth manually |
| Responsible reviewer or monitor | Inspection technician (user) |
| Open question | Does simulated damage help when real damaged examples are few? |

## Next steps

- Damage size: plot detection against cavity diameter (Ø0.10–0.20 mm) to find where detection becomes reliable.
- Check whether flagged segments 18 and 34 come from how teeth are split by angle.
- More healthy revolutions to measure the per-tooth false-alarm rate with more than 17 teeth.
- Real physical data: decide how much synthetic data helps.
- Cross-correlation and the neighbour-tooth healthy reference as a separate experiment.
- Report: PatchCore appears as the inspiration, with one sentence of its results (14/15 at Ø0.20 mm, 6/15 at Ø0.10 mm).
