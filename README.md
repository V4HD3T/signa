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

| model | top-1 | top-5 |
| --- | --- | --- |
| BiLSTM + augmentation | 88.0% | 98.6% |
| **Transformer + augmentation** | **97.1%** | **99.7%** |
| BiLSTM, no augmentation | 88.2% | 98.1% |

Two findings, both measured rather than assumed, and both the reason the
comparison runs exist:

**The Transformer pulls away as the vocabulary grows.** At 26 glosses it led the
BiLSTM by 4.5 points; at 64 it leads by 9. The BiLSTM plateaus around 88% while
the Transformer keeps scaling — the opposite of the usual "attention needs more
data than we have" intuition at this size, and worth saying out loud because the
plan assumed the BiLSTM was the safe baseline and the Transformer a nice-to-have.

**Augmentation stopped mattering for the BiLSTM.** It was worth 4.9 points at 26
glosses and is worth nothing at 64 (88.0 vs 88.2 is noise). More classes meant
more than twice the training clips, and past some point the BiLSTM has enough
real variety that synthetic variety adds nothing — or it has simply hit its
capacity ceiling and augmentation cannot push through it. Either way, "augment
because the dataset is small" is a claim with a shelf life, not a law. It will
be re-measured on AUTSL, which is larger again.

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

Start the MVP at 50–100 of AUTSL's 226 signs, then scale to the full 226 once the
pipeline is honest. `--max-glosses` takes the best-represented classes, so the
headline number never rests on a class with four test clips. Report the full 226
alongside the subset — the leaderboard comparison only means anything at 226.

## Status

- ✅ Frame layout, MediaPipe extraction, normalisation, resampling — 24 tests, no camera or MediaPipe needed to run them
- ✅ Manifest-backed dataset with a validated signer-independent split, scaling to a benchmark's own protocol
- ✅ Landmark-space augmentation (time warp, rotation, scale, jitter; mirroring behind a flag)
- ✅ BiLSTM baseline and Transformer encoder, trained end to end on real video (LSA64), signer-independent
- ✅ Training loop: three-way signer split, early stopping on validation, top-1/top-5 with the protocol recorded in `summary.json`
- ✅ Corpus audit (`signa.audit`) with detection-rate reporting and `--prune`; run orchestration in `scripts/run_lsa64.py`
- ✅ Self-recording tool and push-to-sign webcam demo
- ✅ First measured results: Transformer 97.1% / BiLSTM 88.0% top-1 on LSA64's 64 signs, signer-independent (see Results)
- ⏳ **Next:** AUTSL access via CodaLab registration ([`docs/dataset-access.md`](docs/dataset-access.md)) — the reportable TİD numbers
- ⏳ A confusion matrix over the vocabulary, to see which signs the plateau is made of
- ⏳ (Stretch) Cross-dataset generalisation: train on AUTSL, test on BosphorusSign22k
- ⏳ (Stretch) A thin FastAPI wrapper, so the demo can become a tab in Lingua later

## Roadmap

Ordered so nothing is ever blocked — the slowest gate runs in the background
while real data is already going through the pipeline.

1. **Register on CodaLab for AUTSL**, and mail the BosphorusSign22k EULA request the same day. The second one is a parallel, unbounded wait that costs one email and unlocks the cross-dataset experiment.
2. ✅ **Done:** LSA64 pulled, extracted, audited, and trained end to end. Transformer 97.1% / BiLSTM 88.0% top-1 signer-independent over 64 signs. Not a reportable *TİD* result — Argentinian, gloved signers — but the pipeline works and the model comparison already produced two findings worth keeping (see Results).
3. On AUTSL arrival: `--dry-run`, verify the parse, extract, then train on 50–100 signs with the benchmark's own 31/6/6 split.
4. BiLSTM baseline at the full 226, signer-independent, protocol written down next to the number.
5. Augmentation ablation + Transformer comparison — two measured data points beat one asserted one.
6. Push-to-sign demo over the real vocabulary.
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
  models.py      BiLSTM baseline, Transformer encoder
  train.py       training + signer-independent evaluation
  audit.py       corpus detection-rate report + prune
  demo.py        push-to-sign webcam demo
scripts/
  run_lsa64.py   audit -> prune -> the three comparison runs
tests/           normalisation, resampling, split, augmentation
docs/            dataset access and the EULA request
```
