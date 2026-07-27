# Signa — Isolated Turkish Sign Language Recognition

[![CI](https://github.com/V4HD3T/signa/actions/workflows/ci.yml/badge.svg)](https://github.com/V4HD3T/signa/actions/workflows/ci.yml)

**Version:** 0.1.8 · [changelog](CHANGELOG.md)

Signa recognises isolated Turkish Sign Language (TİD) words from a webcam. Sign a
word — in `--auto` mode it finds the sign's start and end on its own — and the
model returns its top guesses, or an honest "not sure".

Named after the Latin *signa*, "signs", as a sibling to
[Lingua](../lingua) (*lingua*, "tongue"). Same author, same
engineering standards, separate repository: the two share no code and no
deployment, so a `torch` + `mediapipe` dependency tree never lands on Lingua's
API image.

## Scope, stated up front

**In scope:** isolated word recognition over a 50–100 gloss vocabulary, signer-
independent evaluation, a live webcam demo.

**Out of scope:** continuous sign language translation. Recognising connected
signing requires temporal segmentation, handling coarticulation between
adjacent signs, and mapping sign order onto Turkish grammar — it remains an
open research problem, not a graduation project. Signa recognises *words*, one
at a time, and says so everywhere it reports a number.

## Data

Primary dataset is **AUTSL** (Ankara University): 226 signs, 43 signers, 38,336
clips, with an official signer-independent split of 31 / 6 / 6 signers. Full
reasoning, access steps and the two secondary datasets are in
[`docs/dataset-access.md`](docs/dataset-access.md).

The short version: 43 signers is the reason. Signer-independent generalisation is
the whole difficulty of this problem and it is learned from signer diversity —
31 training signers teach a model what is invariant across people in a way that
four cannot.

## Honest accuracy expectations

Published figures on these datasets vary by an enormous margin, and almost all
of the spread is protocol, not modelling. Before quoting any number — in the
report, in the defence, in this README — check three things about it: **how many
classes**, **top-1 or top-5**, and **signer-independent or not**.

The claim that a plain LSTM reaches ~99% here does not survive that check. It is
the shape of a number that is either top-5, or from a split where the same signer
appears in training and test. A realistic target for a plain BiLSTM on landmarks,
evaluated signer-independently, is **75–90% top-1** — higher with fewer classes,
since there is less to confuse. Reported results well above that range generally
combine multiple cues (hand + face + body + optical flow) rather than landmarks
alone.

The AUTSL leaderboard makes this concrete and is worth using rather than
avoiding. The CVPR 2021 ChaLearn challenge winners exceeded 96% — with
multi-stream ensembles over RGB *and* depth *and* pose *and* optical flow. That
is not this project's target; it is what this project positions against:

> *the challenge winner reaches 96% with a four-stream ensemble over RGB and
> depth; this reaches X% from hand and pose landmarks alone, running live on a
> webcam at 30 fps*

That is a better result than a bigger number, because it is true and because it
explains its own gap. Write the protocol down next to every number this repo
produces. "94% top-5, 226 classes, signer-independent" is a defensible sentence.
"99% accuracy" invites one question from the jury — *which split?* — that has no
good answer.

`train.py` reports top-1 and top-5 on a held-out signer, and writes the protocol
into `summary.json` alongside them, so the number and its caveats cannot drift
apart.

## Results so far (LSA64)

The primary target is AUTSL, which is still being obtained. LSA64 is the
plumbing test — Argentinian Sign Language, and the signers wear coloured gloves
MediaPipe was never trained on, so these numbers measure that the pipeline
works, not TİD performance. Reproduce with `python scripts/run_lsa64.py`.

Signer-independent: train on 6 signers, validate on 2 (007/008), test on 2
(009/010). Full 64-sign vocabulary, 625 test clips. Mean either-hand detection
across the corpus is 0.81, so the signing hand is missing from roughly a fifth
of frames before the model sees anything.

| model | top-1 | top-5 | params |
| --- | --- | --- | --- |
| BiLSTM + augmentation | 88.0% | 98.6% | 699k |
| Transformer + augmentation | 97.1% | 99.7% | 424k |
| **TCN + augmentation** | **98.2%** | **100%** | **225k** |
| BiLSTM, no augmentation | 88.2% | 98.1% | 699k |

**Do the pose points earn their place?** The 8 upper-body points (shoulders,
elbows, wrists, hips) sit alongside the hands as input; the design bet was that
hand shape and trajectory carry most of the signal. The ablation (`--no-pose`,
which zeroes the pose block *after* normalisation so the hands stay
position-normalised) measures it: the TCN drops from 98.2% to **96.6% top-1**
without pose. So the bet was right — hands alone already reach 96.6% — but pose is
not free either: it is a real, cheap **+1.6 points**. Both halves of that sentence
matter, and skipping the 468-point face mesh looks well-judged next to what 8
pose points buy.

**Does combining the models help?** Only when they are comparably strong
(`signa.ensemble`, calibrated probabilities averaged). All three together score
97.3% — *below* the TCN's 98.2%, because the BiLSTM at 88% drags the average down.
The two strong models alone (TCN + Transformer) reach 98.6%, a slim +0.3% over the
TCN. The gain is real but small, which fits the earlier finding: the three fail
largely on the same low-detection clips, so there is only so much independent
error for averaging to cancel. Ensembling is not free accuracy here — it helps a
little if you drop the weak member, and hurts if you keep it.

**Is that one number a lucky split?** Leave-one-signer-out cross-validation
(`signa.crossval`) answers it: hold out each of the 10 signers in turn, train on
the other 9, average. The TCN scores **97.7% ± 1.5% top-1** (99.8% ± 0.2% top-5)
across all 10 folds — from 94.7% on the hardest held-out signer to 99.4% on the
easiest. The single-fold 98.2% was mildly optimistic (signer 009 is an easy one),
but the spread is tight: the result barely moves with which signer you hold out,
which is the whole point of reporting it this way. The defensible headline is the
LOSO mean, not the best fold.

Where the errors live (from `signa.report` on the Transformer): the miss is not
spread evenly — one sign sits at 50% while most classes are perfect — and it
tracks hand detection hard. Clips the model got right averaged 0.887 either-hand
detection; clips it got wrong averaged 0.653. Accuracy climbs monotonically with
detection rate, 89.1% in the worst band to 98.4% in the best. A large share of
the residual error is on clips whose features MediaPipe never fully captured —
a fact about the corpus, not the model.

Three findings, all measured rather than assumed, and all the reason the
comparison runs exist:

**Recurrence was the bottleneck, not a lack of attention.** The BiLSTM plateaus
near 88% while the Transformer reached 97%, which looked like a win for attention
— until the TCN, which is neither recurrent nor attentional, edged *past* the
Transformer at 98.2% with roughly a third of the BiLSTM's parameters. What the
BiLSTM lacked was not attention but parallel access to the whole clip; a
sequential hidden-state bottleneck is what held it back. Both convolution and
attention clear it, and the convolution does so more cheaply.

**The Transformer's lead over the BiLSTM widened with the vocabulary** — 4.5
points at 26 glosses, 9 at 64 — the opposite of the usual "attention is
data-hungry" intuition at ~30 clips per class. The plan treated the BiLSTM as
the safe baseline; on this data it is the weakest of the three.

**Augmentation stopped mattering for the BiLSTM.** It was worth 4.9 points at 26
glosses and nothing at 64 (88.0 vs 88.2 is noise). More classes meant more than
twice the training clips, and past some point real variety crowds out synthetic
variety — or the BiLSTM has simply hit its ceiling. "Augment because the dataset
is small" is a claim with a shelf life, not a law. Re-measured on AUTSL, which is
larger again.

## How it works

```
webcam / clip ──▶ MediaPipe Holistic ──▶ 152-d frame ──▶ normalise ──▶ resample ──▶ BiLSTM ──▶ gloss
                                        (both hands +     (shoulder-    (48 frames)
                                         8 pose points +    centred,
                                         presence flags)    shoulder-
                                                            scaled)
```

**Features.** Both hands (21×3 each) plus eight upper-body pose points
(shoulders, elbows, wrists, hips), and two flags marking whether each hand was
detected at all. The 468-point face mesh is skipped: in isolated signing the
signal is hand shape and trajectory, and at ~20 training clips per gloss those
extra points are 1400 dimensions of noise.

**Normalisation.** Per frame, the origin moves to the shoulder midpoint and
everything is divided by shoulder width. A signer standing further from the
camera, or off to one side, produces the same vectors.

**Resampling, not padding.** Every clip becomes exactly 48 frames by linear
interpolation. Padding would make clip *duration* a usable feature, which fails
precisely where it matters — on an unseen signer with a different tempo.

**Landmarks, not pixels.** Training is cheap: this runs on a CPU or Colab's free
tier. Extraction is the expensive step, and it happens once.

**Raw on disk, normalised at load.** Extraction writes raw MediaPipe output;
normalisation and resampling happen in the `Dataset`. The normaliser can be
changed without re-running MediaPipe over 20k videos.

## Evaluation protocol

Three-way signer split. On AUTSL this is the benchmark's own protocol, passed
explicitly rather than reinvented:

| Split | AUTSL signers | Used for |
| --- | --- | --- |
| Train | 31 | fitting |
| Validation | 6 | early stopping, checkpoint selection |
| Test | 6 | the reported number, touched once |

```bash
python -m signa.train --test-signers <the 6> --val-signers <the 6>
```

The validation signers are not ceremony. Selecting a checkpoint on the test
signers turns the headline into a best-of-N over the test set — the same
overfitting the signer-independent split exists to prevent, one level up. Left to
its own devices `train.py` holds out as many validation signers as there are test
signers, so validation is never noisier than the number it is chasing.

Augmentation — time warping (±20%), in-plane rotation (±8°), scaling (±10%) and
coordinate jitter, all in shoulder-width units — matters most on the smaller
datasets, where a gloss may have only ~20 training clips. Mirroring is
implemented but off by default: it models a left-handed signer, but it also
destroys the dominant/non-dominant asymmetry that separates some sign pairs.
Measure it before enabling it.

## Setup

> **Python 3.11 or 3.12.** MediaPipe publishes no wheel for 3.13+, and the
> system interpreter here is 3.14. The venv has to be an older interpreter —
> everything else in the stack is fine on 3.14, but `pip install mediapipe`
> will simply fail to resolve.

> **Keep the checkout on an ASCII-only path.** On Windows MediaPipe resolves its
> graph assets (`holistic_landmark_cpu.binarypb`) through a C++ layer that
> mangles non-ASCII characters in the path. This project started life in a
> directory called `Yapay Zeka İşaret Dili Tercümanı`, and every `Holistic()`
> construction died with a `FileNotFoundError` quoting a corrupted path —
> nothing in the message points at the folder name. That is why the directory is
> called `signa`. Do not rename it back to anything with Turkish characters,
> however much more descriptive that would be.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Tests run without MediaPipe or a camera:

```bash
pytest
```

## Commands

Record your own clips (works today, no dataset needed):

```bash
python -m signa.record --gloss merhaba --signer me --count 5
```

Check filename parsing before spending hours on extraction:

```bash
python -m signa.extract --videos data/raw --dry-run
```

Extract landmarks and write the manifest:

```bash
python -m signa.extract --videos data/raw --out data/landmarks --stride 2 --workers 12
```

`--stride 2` keeps every second frame, which turns a 60 fps source into the
30 fps the other datasets use — a model should not have to relearn tempo per
corpus — and halves the work. `--workers` runs one MediaPipe graph per process;
at 1080p this is the difference between six hours and fifteen minutes. Extraction
resumes: clips already on disk are skipped, so an interrupted run costs nothing.

Train and evaluate signer-independently:

```bash
python -m signa.train --model bilstm --max-glosses 50 --test-signers User_1
```

Cross-validate over every signer (the defensible headline, not one fold):

```bash
python -m signa.crossval --manifest data/manifest_train_full.csv \
    --landmark-root data/landmarks_lsa64 --model tcn --tag lsa64
```

Gather every run into one table (the numbers above come from here):

```bash
python -m signa.results --markdown
```

Compare against the Transformer:

```bash
python -m signa.train --model transformer --max-glosses 50 --tag transformer
```

Analyse what a trained model gets wrong — per-class accuracy, confused pairs,
and accuracy by hand-detection rate:

```bash
python -m signa.report --checkpoint runs/lsa64-full-transformer/best.pt \
    --manifest data/manifest_train_full.csv --landmark-root data/landmarks_lsa64 \
    --test-signers 009 010 --val-signers 007 008 --csv confusion.csv
```

Calibrate a "not sure" reject option (writes a sidecar the demo and tutor use):

```bash
python -m signa.reject --checkpoint runs/lsa64-full-transformer/best.pt \
    --manifest data/manifest_train_full.csv --landmark-root data/landmarks_lsa64 \
    --test-signers 009 010 --val-signers 007 008 --target-accuracy 0.99 --write
```

Run the live demo — hold SPACE, or `--auto` to sign continuously with no key:

```bash
python -m signa.demo --checkpoint runs/lsa64-full-tcn/best.pt --auto
```

Practise signing with the trained model as a tutor (also takes `--auto`):

```bash
python -m signa.learn --checkpoint runs/lsa64-full-tcn/best.pt --labels names.json
```

## Knowing when to say nothing

A softmax over 64 classes always sums to one, so *something* always wins — even
for a fumble, noise, or a sign the model was never taught. `signa.reject` lets it
decline, in two measured steps:

**Calibrate.** Temperature scaling (Guo et al. 2017) fits one scalar T on the
validation signers to make the reported confidence honest — it cannot change
which sign wins, only how confident the model sounds. On the LSA64 Transformer
T fitted to 0.775 (below 1: this model was mildly *under*confident, the opposite
of the usual case, which is why it is fitted and not assumed).

**Reject.** Below a confidence threshold — chosen on validation for a target
accuracy among accepted signs, not a guessed 0.5 — the prediction is withheld as
"not sure". On the test signers, rejecting the least-confident 5.3% of signs
lifts accuracy on the rest from 97.1% to 98.5%. The withheld signs are
disproportionately the ones the model would have got wrong, consistent with the
detection-rate finding above.

Calibration is fitted on validation and reported on test — the same discipline
as the split, so the reject option is never tuned on the data it is judged on. It
writes a sidecar next to the checkpoint; the demo shows "not sure" below the
threshold, and learning mode treats an unreadable attempt as "try again" rather
than scoring it as a forgotten sign.

## Learning mode

The recogniser turned around: instead of the user signing and the model
guessing, the app names a sign, the user performs it, and the model grades the
attempt. That grade drives a spaced-repetition schedule (SM-2), so signs a
learner struggles with come back soon and mastered ones recede — the pedagogy
Lingua runs for vocabulary, here for signing, which is what makes Signa a thing
you *use* rather than a model you evaluate.

All the judgement lives in `signa.practice`, hardware-free and tested: an attempt
grades to **correct / close / missed** (close = the right sign was in the model's
top-k but not first — a beginner whose sign is readable but not crisp is in a
different place from one whose sign was not read at all), an SM-2 card advances
(1 → 6 → interval×ease days; a lapse resets and the ease factor floors at 1.3 so
a hard sign never disappears), and the scheduler clears due backlog before
introducing new material. Streak and daily goal are counted in the learner's
local dates, not UTC. Progress persists to JSON between sessions.

Gloss ids like `001` are not words; pass `--labels names.json` (a `{gloss: name}`
map) to show real sign names.

## Live demo: two ways to bracket a sign

The model is trained on pre-trimmed clips, so "when does a sign start and stop"
is a *different* problem — segmentation, the same one that makes continuous
translation hard.

**Push-to-sign** (default) sidesteps it: hold SPACE, sign, release. Reliable, and
the honest fallback.

**`--auto`** solves it. `signa.segment` watches motion energy — per-frame hand
velocity in shoulder-width units — and a two-state machine with hysteresis finds
each sign's start and end, so nothing is held. It takes more motion to start a
sign than to keep one going, so a dip mid-sign doesn't cut it in two; a minimum
length rejects twitches, a maximum force-cuts a signer who never rests. A lost
hand is treated as "hold", not stillness — this corpus drops a hand on a fifth of
frames, and without that distinction detection gaps masquerade as sign endings.
The finite-state machine is pure and tested on synthetic energy streams; the
webcam loop only feeds it frames. Segmentation quality is bounded by
hand-detection quality, which is why push-to-sign stays the default.

## Vocabulary choice

Start the MVP at 50–100 of AUTSL's 226 signs, then scale to the full 226 once the
pipeline is honest. `--max-glosses` takes the best-represented classes, so the
headline number never rests on a class with four test clips. Report the full 226
alongside the subset — the leaderboard comparison only means anything at 226.

## Status

- ✅ Frame layout, MediaPipe extraction, normalisation, resampling — 24 tests, no camera or MediaPipe needed to run them
- ✅ Manifest-backed dataset with a validated signer-independent split, scaling to a benchmark's own protocol
- ✅ Landmark-space augmentation (time warp, rotation, scale, jitter; mirroring behind a flag)
- ✅ Three architectures — BiLSTM, Transformer, TCN — trained end to end on real video (LSA64), signer-independent; the TCN wins at 98.2% top-1 on a third of the BiLSTM's parameters
- ✅ Training loop: three-way signer split, early stopping on validation, top-1/top-5 with the protocol recorded in `summary.json`
- ✅ Corpus audit (`signa.audit`) with detection-rate reporting and `--prune`; run orchestration in `scripts/run_lsa64.py`
- ✅ Self-recording tool and push-to-sign webcam demo
- ✅ First measured results: Transformer 97.1% / BiLSTM 88.0% top-1 on LSA64's 64 signs, signer-independent (see Results)
- ✅ Error analysis (`signa.report`): per-class accuracy, confused pairs, and accuracy-by-detection-rate — the errors track hand detection, not just the model
- ✅ Learning mode (`signa.learn` + `signa.practice`): SM-2 spaced repetition, grading, streaks and daily goal — pedagogy tested hardware-free
- ✅ Calibrated reject option (`signa.reject`): temperature scaling + a validation-chosen threshold — rejecting 5.3% of signs lifts accuracy on the rest from 97.1% to 98.5%
- ✅ Automatic segmentation (`signa.segment`): motion-energy FSM with hysteresis and dropout handling, keyless `--auto` mode in demo and tutor — push-to-sign removed as a requirement
- ✅ Leave-one-signer-out cross-validation (`signa.crossval`): TCN 97.7% ± 1.5% top-1 over all 10 folds — the headline no longer rests on one lucky split
- ✅ Model ensemble (`signa.ensemble`): calibrated soft/hard voting — the two strong models reach 98.6%, but adding the weak one hurts; a measured "barely helps"
- ✅ Pose ablation (`--no-pose`): the 8 pose points are worth +1.6 top-1 (98.2% → 96.6%) — hands carry most of it, pose adds a real little
- ✅ CI (`.github/workflows/ci.yml`): the 126-test suite on Python 3.11 and 3.12, every push
- ✅ Results aggregation (`signa.results`): every run's numbers in one table, read from the artifacts rather than copied by hand
- ⏳ **Next:** AUTSL access via CodaLab registration ([`docs/dataset-access.md`](docs/dataset-access.md)) — the reportable TİD numbers
- ⏳ (Stretch) Cross-dataset generalisation: train on AUTSL, test on BosphorusSign22k
- ⏳ (Stretch) A thin FastAPI wrapper, so the demo can become a tab in Lingua later

## Roadmap

Ordered so nothing is ever blocked — the slowest gate runs in the background
while real data is already going through the pipeline.

1. **Register on CodaLab for AUTSL**, and mail the BosphorusSign22k EULA request the same day. The second one is a parallel, unbounded wait that costs one email and unlocks the cross-dataset experiment.
2. ✅ **Done:** LSA64 pulled, extracted, audited, and trained end to end. Three architectures compared (TCN 98.2% / Transformer 97.1% / BiLSTM 88.0% top-1, signer-independent over 64 signs), a calibrated reject option, learning mode, and `--auto` segmentation. Not a reportable *TİD* result — Argentinian, gloved signers — but the whole pipeline works, keyless, with findings worth keeping (see Results).
3. On AUTSL arrival: `--dry-run`, verify the parse, extract, then train on 50–100 signs with the benchmark's own 31/6/6 split.
4. All three models at the full 226, signer-independent, protocol written down next to each number.
5. Re-measure the augmentation ablation and the architecture comparison on real TİD data — do the LSA64 findings hold?
6. Keyless demo and tutor over the real vocabulary.
7. (Stretch) Cross-dataset generalisation, then the HTTP boundary and a Lingua tab.

## Layout

```
src/signa/
  config.py      frame layout + run configuration
  landmarks.py   MediaPipe -> frames; normalise; resample   (shared by train and demo)
  extract.py     videos -> landmark .npy + manifest.csv
  record.py      webcam -> clips, for testing without the dataset
  dataset.py     manifest, signer-independent split, torch Dataset
  augment.py     landmark-space augmentation
  models.py      BiLSTM baseline, Transformer encoder, TCN
  train.py       training + signer-independent evaluation
  crossval.py    leave-one-signer-out cross-validation over train.run
  results.py     gather every run's summary into one table   (no hardware)
  audit.py       corpus detection-rate report + prune
  report.py      per-class accuracy, confused pairs, error vs detection rate
  reject.py      temperature calibration + "not sure" reject option   (no hardware)
  ensemble.py    combine models: calibrated soft/hard voting   (no hardware)
  segment.py     motion-energy segmentation FSM + frame buffer   (no hardware)
  demo.py        webcam demo: push-to-sign, or --auto continuous
  practice.py    learning-mode pedagogy: grading, SM-2, streaks   (no hardware)
  learn.py       learning-mode webcam session (push-to-sign or --auto)
scripts/
  run_lsa64.py   audit -> prune -> the three comparison runs
tests/           normalisation, resampling, split, augmentation
docs/            dataset access and the EULA request
```
