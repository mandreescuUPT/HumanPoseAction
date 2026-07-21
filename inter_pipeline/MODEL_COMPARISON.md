# Model comparison: pretrained NTU checkpoints vs. our MediaPipe-trained model

**Date:** 2026-07-18
**Test set:** 180 clips from `nturgb+d_rgb_2` (NTU setup **S002**)
**Models:** `st_gcn.ntu-xsub`, `st_gcn.ntu-xview` (pretrained, Kinect 3D) vs.
`humanpose5_coco.pth` (ours, trained on MediaPipe 2D — see [TRAINING.md](TRAINING.md))

---

## Test set design

| Property | Value |
|---|---|
| Source | `nturgb+d_rgb_2` — NTU **setup S002** |
| Subjects | **P009–P014** (6 people) |
| Actions | A008 sitting down, A009 standing up, A023 hand waving, A027 jump up, A043 falling |
| Cameras | C001, C002, C003 (all three) |
| Repetitions | R001, R002 |
| **Total** | **180 clips — exactly 36 per action** |

**Why these 180 clips?** Three deliberate choices:

1. **Truly unseen subjects.** Our model trained on setup S001, subjects
   P001/P003–P008. S002 also contains P003, P007, P008 — the *same people* —
   so those were **excluded**. P009–P014 appear nowhere in training.
2. **A different setup entirely.** S002 is a different physical camera
   arrangement from the S001 data the model trained on. So this measures
   generalisation across both *people* and *recording conditions* — a
   substantially harder and more honest test than the original 20-clip P002
   validation set.
3. **Balanced and large enough to trust.** 36 clips per class means no class
   can inflate the average. At n=180, a 91% result carries a 95% confidence
   interval of roughly **±4%**; the old 20-clip val set gave ±15–20%, which was
   too wide to draw conclusions from. All 3 cameras are included, which also
   exercises cross-view robustness — directly relevant since `ntu-xview` is
   under test.

---

## Headline results

Two scorings are reported, because the models don't natively make the same
decision — the pretrained models choose among **60** NTU actions, ours among
**5**. Comparing those directly would be unfair to the pretrained models, so the
**restricted** column constrains every model to the same 5-way choice.

| Model | Open (60-way) top-1 | **Restricted (5-way) top-1** | Mean GT rank (of 60) |
|---|---|---|---|
| `ntu-xsub` (pretrained) | 7/180 — **3.9%** | 70/180 — **38.9%** | 19.4 |
| `ntu-xview` (pretrained) | 9/180 — **5.0%** | 46/180 — **25.6%** | 27.2 |
| **`humanpose5_coco` (ours)** | 164/180 — **91.1%** | 164/180 — **91.1%** | 1.12 |

Random-guess baselines: **1.7%** (60-way), **20%** (5-way).

**The fair comparison is the restricted column**, and it is decisive: given the
identical 5-way decision on identical input, our model reaches **91.1%** while
the pretrained checkpoints manage **38.9%** and **25.6%**. `ntu-xview` is barely
above the 20% random baseline; `ntu-xsub` roughly doubles it. Both are far below
usable.

---

## Per-class breakdown (restricted 5-way)

| Action | `ntu-xsub` | `ntu-xview` | **ours** |
|---|---|---|---|
| sitting down | 0/36 (0%) | 0/36 (0%) | **30/36 (83%)** |
| standing up | 0/36 (0%) | 0/36 (0%) | **32/36 (89%)** |
| hand waving | 3/36 (8%) | 0/36 (0%) | **36/36 (100%)** |
| jump up | 4/36 (11%) | 9/36 (25%) | **36/36 (100%)** |
| falling | 0/36 (0%) | 0/36 (0%) | **30/36 (83%)** |

The pretrained models score **exactly 0%** on three of five actions. This is
worse than random guessing would achieve and reveals the failure isn't
"somewhat degraded accuracy" — the models are not tracking these actions at all.
Their few successes cluster on `jump up`, plausibly because a whole-body
vertical translation survives the 2D/3D mismatch better than posture changes do.

## Confusion matrix — our model

Rows = ground truth, columns = predicted:

| | sitting down | standing up | hand waving | jump up | falling |
|---|---|---|---|---|---|
| **sitting down** | **30** | 4 | 2 | 0 | 0 |
| **standing up** | 0 | **32** | 4 | 0 | 0 |
| **hand waving** | 0 | 0 | **36** | 0 | 0 |
| **jump up** | 0 | 0 | 0 | **36** | 0 |
| **falling** | 5 | 0 | 0 | 1 | **30** |

Every error is intuitively explainable rather than random:

- **falling → sitting down (5)** — the largest confusion, and unsurprising: both
  are downward whole-body motions ending near the floor.
- **sitting down ↔ standing up (4)** — these are near time-reverses of each
  other, traversing almost identical joint positions in opposite order.
- **standing up → hand waving (4)** — likely clips where arm motion dominates
  the visible signal.
- `hand waving` and `jump up` are **perfect (36/36)** — the two most kinematically
  distinctive actions in the set.

The error structure is a model making *sensible* mistakes on genuinely similar
motions, not one guessing.

---

## Interpretation

**1. This confirms the 2D/3D incompatibility conclusively, on a large sample.**
The pretrained checkpoints were trained on ~40,000 clips of Kinect 3D skeleton
data and report 81.5% / 88.3% top-1 in the original ST-GCN paper. Here they score
3.9% / 5.0% open, and 38.9% / 25.6% even when handed a 5-way choice. The models
are not defective — they are being fed a data type they were never built for.
MediaPipe estimates 2D image coordinates from RGB; Kinect measures true metric 3D
depth. No conversion bridges that, which is exactly why this pipeline exists.

**2. Our model wins by a very large margin — ~2.3× the better pretrained model.**
91.1% vs 38.9% on the same decision, same clips, same input format. Non-overlapping
confidence intervals; the gap is far beyond sampling noise.

**3. It generalises, which is the important part.** The 91.1% is on six people the
model has never seen, in a different recording setup than it trained in. It is not
memorisation. Notably, this is *close to* the 95% measured on the original 20-clip
P002 validation set — the small drop (95% → 91%) is what you'd expect from a
harder, more honest test, and it confirms that the earlier number was not a fluke
of a thin sample.

**4. Training on 149 clips beat models trained on 40,000.** Not because the
architecture is better — it's the identical ST-GCN — but because the training data
matched the deployment data. Data *fit* dominated data *volume* by a wide margin.

## Limitations

- **Only 5 actions.** This does not contradict the earlier ~15% result on all 60
  NTU classes. Those 60 include ~11 two-person actions MediaPipe structurally
  cannot capture (it tracks one person) and many near-identical pairs. The
  finding is that a *well-chosen, distinct, single-person* class set works — not
  that 60-way is solved.
- **Still only 8 training subjects.** All from setup S001. Subject diversity
  remains the main ceiling on extending the class count.
- **Top-5 is meaningless for a 5-class model** (trivially 100%) — top-1 is the
  only meaningful metric here, which is what's reported throughout.
- **Single trained model.** Numbers come from one training run with one
  train/val split. Leave-one-subject-out cross-validation would tighten the
  estimate further.

## Reproducing

```powershell
cd C:\Users\alext\Desktop\IPCEI\pipeline

# 180-clip evaluation across all three models
$clips = Get-ChildItem ..\nturgb+d_rgb_2\nturgb+d_rgb\*.avi |
    Where-Object { $_.Name -match 'P(009|010|011|012|013|014)' -and
                   $_.Name -match 'A(008|009|023|027|043)' }
..\env\Scripts\python.exe run_pipeline.py $clips.FullName `
    --model ntu-xsub ntu-xview `
    --custom-model models\humanpose5_coco.pth `
    --results-dir results_s002
```

Per-clip reports: `results_s002/*.json`; aggregate: `results_s002/summary.json`.


Example inference :
PS C:\Users\alext\Desktop\IPCEI\pipeline> python .\run_pipeline.py "C:\Users\alext\Desktop\nturgb+d_rgb\S003C001P001R001A008_rgb.avi" --model all --results-dir results_s002        
[stage1] keypoints already exist, skipping extraction: C:\Users\alext\Desktop\IPCEI\HumanPoseAction\output\S003C001P001R001A008_rgb_keypoints_full_body.json
[infer] detected HumanPoseAction export -> converting to layout 'ntu-rgb+d'
[infer] clip='S003C001P001R001A008_rgb'  layout=ntu-rgb+d  V=25  frames=76 (detected=76)
[infer] model input tensor (N,C,T,V,M) = (1, 3, 76, 25, 2)
[infer] detected HumanPoseAction export -> converting to layout 'ntu-rgb+d'
[infer] clip='S003C001P001R001A008_rgb'  layout=ntu-rgb+d  V=25  frames=76 (detected=76)
[infer] model input tensor (N,C,T,V,M) = (1, 3, 76, 25, 2)
[infer] detected HumanPoseAction export -> converting to layout 'openpose'
[infer] clip='S003C001P001R001A008_rgb'  layout=openpose  V=18  frames=76 (detected=76)
[infer] model input tensor (N,C,T,V,M) = (1, 3, 76, 18, 2)

==========================================================================
  clip:          S003C001P001R001A008_rgb
  frames:        76/76 detected
  ground truth:  sitting down  (A008)
--------------------------------------------------------------------------
  ntu-xsub  [ntu-rgb+d]  ->  miss (GT #31)
     1. touch back (backache)                         89.9%
     2. wear a shoe                                    2.9%
     3. touch chest (stomachache/heart pain)           2.7%
     4. take off a shoe                                1.4%
     5. kicking something                              1.1%
--------------------------------------------------------------------------
  ntu-xview  [ntu-rgb+d]  ->  miss (GT #48)
     1. standing up                                   60.2%
     2. typing on a keyboard                          14.7%
     3. touch chest (stomachache/heart pain)          11.1%
     4. touch back (backache)                          7.3%
     5. reach into pocket                              2.7%
--------------------------------------------------------------------------
  kinetics  [openpose]  ->  unscored: different label space
     1. lunge                                         69.9%
     2. catching or throwing baseball                  9.3%
     3. catching or throwing softball                  7.4%
     4. throwing ball                                  2.8%
     5. squat                                          2.4%
==========================================================================

[main] report -> results_s002\S003C001P001R001A008_rgb.json


Example code 

PS C:\Users\alext\Desktop\IPCEI\pipeline> python .\run_pipeline.py "C:\Users\alext\Desktop\nturgb+d_rgb\S003C001P001R001A008_rgb.avi" --model ntu-xsub ntu-xview --custom-model models\humanpose5_coco.pth  --results-dir results_s002
[stage1] extracting keypoints with: C:\Users\alext\AppData\Local\Programs\Python\Python311\python.exe

=======================================================
  Pose Detector — mode: BODY
  Source:  C:\Users\alext\Desktop\nturgb+d_rgb\S003C001P001R001A008_rgb.avi
  Frames: 76
  Original res: 1920x1080 @ 30.0 fps
  Max frame size: 600 px
  Output: C:\Users\alext\Desktop\IPCEI\HumanPoseAction\output
=======================================================

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

=======================================================
  Done! Frames processed: 76
  Total detections:        76
  JSON saved:            C:\Users\alext\Desktop\IPCEI\HumanPoseAction\output\S003C001P001R001A008_rgb_keypoints_full_body.json
=======================================================

[stage1] keypoints -> C:\Users\alext\Desktop\IPCEI\HumanPoseAction\output\S003C001P001R001A008_rgb_keypoints_full_body.json
[infer] detected HumanPoseAction export -> converting to layout 'ntu-rgb+d'
[infer] clip='S003C001P001R001A008_rgb'  layout=ntu-rgb+d  V=25  frames=76 (detected=76)
[infer] model input tensor (N,C,T,V,M) = (1, 3, 76, 25, 2)
[infer] detected HumanPoseAction export -> converting to layout 'ntu-rgb+d'
[infer] clip='S003C001P001R001A008_rgb'  layout=ntu-rgb+d  V=25  frames=76 (detected=76)
[infer] model input tensor (N,C,T,V,M) = (1, 3, 76, 25, 2)
[infer] detected HumanPoseAction export -> converting to layout 'coco'
[infer] clip='S003C001P001R001A008_rgb'  layout=coco  V=17  frames=76 (detected=76)
[infer] model input tensor (N,C,T,V,M) = (1, 3, 76, 17, 2)

==========================================================================
  clip:          S003C001P001R001A008_rgb
  frames:        76/76 detected
  ground truth:  sitting down  (A008)
--------------------------------------------------------------------------
  ntu-xsub  [ntu-rgb+d]  ->  miss (GT #31)
     1. touch back (backache)                         89.9%
     2. wear a shoe                                    2.9%
     3. touch chest (stomachache/heart pain)           2.7%
     4. take off a shoe                                1.4%
     5. kicking something                              1.1%
--------------------------------------------------------------------------
  ntu-xview  [ntu-rgb+d]  ->  miss (GT #48)
     1. standing up                                   60.2%
     2. typing on a keyboard                          14.7%
     3. touch chest (stomachache/heart pain)          11.1%
     4. touch back (backache)                          7.3%
     5. reach into pocket                              2.7%
--------------------------------------------------------------------------
  humanpose5_coco  [coco]  ->  CORRECT
     1. sitting down                                  98.1% <-- ground truth
     2. hand waving                                    1.8%
     3. falling                                        0.1%
     4. standing up                                    0.0%
     5. jump up                                        0.0%
==========================================================================

[main] report -> results_s002\S003C001P001R001A008_rgb.json