# Signa — Isolated Turkish Sign Language Recognition

**Version:** 0.0.1

Signa recognises isolated Turkish Sign Language (TİD) words from a webcam. Hold
a key, sign one word, release — the model returns its top guesses.

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

## Honest accuracy expectations

Published figures on BosphorusSign22k vary by an enormous margin, and almost all
of the spread is protocol, not modelling. Before quoting any number — in the
report, in the defence, in this README — check three things about it: **which
subset** (50 / 174 / 744 glosses), **top-1 or top-5**, and **signer-independent
or not**.

The claim that a plain LSTM reaches ~99% on this dataset does not survive that
check. It is the shape of a number that is either top-5, or from a split where
the same signer appears in training and test. A realistic target for a plain
BiLSTM on landmarks, evaluated signer-independently, is **75–90% top-1** — higher
at 50 glosses than at 174, since there are fewer classes to confuse. Reported
results well above that range generally combine multiple cues (hand + face +
body + optical flow) rather than landmarks alone.

Write down the protocol next to every number this repo produces. "94% top-5,
174 glosses, signer-independent" is a defensible sentence. "99% accuracy" invites
one question from the jury — *which split?* — that has no good answer.

`train.py` reports top-1 and top-5 on a held-out signer, and writes the protocol
into `summary.json` alongside them, so the number and its caveats cannot drift
apart.

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

Three-way signer split, following the dataset authors' own recommendation:

| Split | Signers | Used for |
| --- | --- | --- |
| Train | 4 | fitting |
| Validation | 1 | early stopping, checkpoint selection |
| Test | 1 | the reported number, touched once |

The extra validation signer is not ceremony. Selecting a checkpoint on the test
signer turns the headline into a best-of-N over the test set — the same
overfitting the signer-independent split exists to prevent, one level up.

At ~5 repetitions × 4 training signers, a gloss has roughly 20 training clips.
At that size augmentation is not optional: time warping (±20%), in-plane
rotation (±8°), scaling (±10%), and coordinate jitter, all in shoulder-width
units. Mirroring is implemented but off by default — it models a left-handed
signer, but it also destroys the dominant/non-dominant asymmetry that separates
some gloss pairs. Measure it before enabling it.

## Setup

> **Python 3.11 or 3.12.** MediaPipe publishes no wheel for 3.13+, and the
> system interpreter here is 3.14. The venv has to be an older interpreter —
> everything else in the stack is fine on 3.14, but `pip install mediapipe`
> will simply fail to resolve.

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
python -m signa.extract --videos data/raw --out data/landmarks
```

Train and evaluate signer-independently:

```bash
python -m signa.train --model bilstm --max-glosses 50 --test-signers User_1
```

Compare against the Transformer:

```bash
python -m signa.train --model transformer --max-glosses 50 --tag transformer
```

Run the live demo:

```bash
python -m signa.demo --checkpoint runs/baseline-bilstm/best.pt
```

## Live demo: push-to-sign

The model is trained on pre-trimmed clips, so "when does a sign start and stop"
is a *different* problem — segmentation, the same one that makes continuous
translation hard. Holding a key while signing removes it from the MVP entirely
rather than solving it badly. Automatic motion-energy segmentation is a v2 item,
and is where the interesting failure modes live.

## Vocabulary choice

Pick the MVP's 50–100 glosses from the dataset's **174-gloss general/daily
subset**, not at random across all 744. Two reasons: results stay directly
comparable to published baselines, and the established signer-independent split
comes for free. `--max-glosses` takes the best-represented glosses, so the
headline number does not rest on classes with four test clips.

## Status

- ✅ Frame layout, MediaPipe extraction, normalisation, resampling — 24 tests, no camera or MediaPipe needed to run them
- ✅ Manifest-backed dataset with a validated signer-independent split
- ✅ Landmark-space augmentation (time warp, rotation, scale, jitter; mirroring behind a flag)
- ✅ BiLSTM baseline and Transformer encoder, both verified end to end on synthetic data
- ✅ Training loop: three-way signer split, early stopping on validation, top-1/top-5 with the protocol recorded in `summary.json`
- ✅ Self-recording tool and push-to-sign webcam demo
- ⏳ **Blocked on data:** dataset access (see [`docs/dataset-access.md`](docs/dataset-access.md) — send the EULA request today)
- ⏳ Real accuracy numbers, a confusion matrix over the MVP vocabulary, and the BiLSTM/Transformer comparison
- ⏳ (Stretch) A thin FastAPI wrapper, so the demo can become a tab in Lingua later

## Roadmap

1. **Send the EULA request** — the only step with an unbounded, external wait
2. Meanwhile: record own clips, run the full pipeline on them, fix whatever the exercise breaks
3. On arrival: `--dry-run`, verify the parse, extract 50–100 glosses from the 174 subset
4. BiLSTM baseline, signer-independent, with the protocol written down
5. Augmentation ablation + Transformer comparison — two measured data points beat one asserted one
6. Push-to-sign demo over the real vocabulary
7. (Stretch) HTTP boundary, then a Lingua tab

## Layout

```
src/signa/
  config.py      frame layout + run configuration
  landmarks.py   MediaPipe -> frames; normalise; resample   (shared by train and demo)
  extract.py     videos -> landmark .npy + manifest.csv
  record.py      webcam -> clips, for testing without the dataset
  dataset.py     manifest, signer-independent split, torch Dataset
  augment.py     landmark-space augmentation
  models.py      BiLSTM baseline, Transformer encoder
  train.py       training + signer-independent evaluation
  demo.py        push-to-sign webcam demo
tests/           normalisation, resampling, split, augmentation
docs/            dataset access and the EULA request
```
