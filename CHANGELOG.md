# Changelog

One entry per version. The rule Signa borrows from Lingua: an entry says what
changed *and why it was worth doing*, so the reasoning behind a decision
survives longer than the diff. Newest first.

Versioning is `0.0.x` while the pipeline is proven on plumbing-test data
(LSA64). The jump to **0.1.0** is reserved for automatic segmentation — the
feature that removes push-to-sign and lets the demo find sign boundaries on its
own, which is the point Signa stops being an isolated-clip classifier and starts
being something you can sign at continuously.

## 0.0.3 — learning mode

The recogniser, flipped into a tutor. Instead of the user signing and the model
guessing, `signa.learn` names a sign, the user performs it, and the model grades
the attempt — which then drives a spaced-repetition schedule, so signs a learner
struggles with recur soon and mastered ones recede. This is the pedagogy Lingua
runs for vocabulary, ported to signing, and it is the feature that makes Signa a
thing you *use* rather than a classifier you evaluate.

The pedagogy is all in `signa.practice`, with no camera and no torch: grade an
attempt into one of correct / close / missed (close = the right sign was in the
model's top-k but not first, a genuinely different place from unrecognised);
advance an SM-2 card (1 → 6 → interval×ease days, a lapse resets and the ease
factor floors at 1.3 so a hard sign never vanishes into a months-long gap);
choose what to practise next (due backlog before new material, most-overdue
first); count a streak and a daily goal in the learner's *local* dates, the same
UTC-rollover lesson Lingua learned. `signa.learn` is the webcam loop around it,
reusing the demo's checkpoint loading and classifier so training, demo, and
tutor read landmarks identically. Progress persists to JSON between sessions.

Gloss ids like `001` are not words, so `--labels names.json` maps them to real
sign names for display; without it the raw id is shown.

20 new tests pin every transition — grading buckets, SM-2 steps and lapses, the
ease floor, scheduler priority, streak edges (yesterday still counts, a two-day
gap does not), and the store round-trip. 51 tests total.

## 0.0.2 — what the model gets wrong, not just how often

`signa.report`: per-class accuracy, the most-confused gloss pairs, and — the
part the corpus audit was building toward — accuracy binned by hand-detection
rate. A single top-1 number cannot say whether the error is spread evenly or
piled onto a few sign pairs, nor whether the model is failing where the hands
were visible or where MediaPipe never found them. On the LSA64 Transformer both
answers turned out to matter: the 2.9% error is concentrated (one sign at 50%,
most classes perfect), and it tracks detection rate hard — clips the model got
right averaged 0.887 either-hand detection, clips it got wrong averaged 0.653.
A large share of the residual error is on clips whose features were never there
to classify, which is a finding about the *corpus*, not the model.

The signer split that both training and this report depend on is now defined
once, in `dataset.make_splits`, rather than reconstructed at each call site. The
trustworthiness of every reported number rests on that split; a second, subtly
different copy of it in the reporting path was a latent way for "test" to stop
meaning what the headline number meant.

Establishes this changelog and the versioning scheme above.

## 0.0.1 — the pipeline, end to end, on real video

Initial scaffold and first real results.

- **Landmark pipeline**, shared byte-for-byte between training and the live
  demo: MediaPipe Holistic → a 152-d frame (both hands, eight upper-body pose
  points, two presence flags) → per-frame shoulder normalisation → resample to
  48 frames. Raw MediaPipe output on disk, normalisation at load, so the
  normaliser can change without re-extracting.
- **Signer-independent evaluation**, three-way: train / validation / test
  signers all disjoint, validation scaled to the test set so checkpoint
  selection is never made on a noisier sample than the number it chases.
- **BiLSTM baseline and Transformer encoder**, plus landmark-space augmentation
  (time warp, rotation, scale, jitter; mirroring behind a flag).
- **Parallel, resumable extraction** (`--workers`, `--stride`) and a corpus
  audit (`signa.audit`, with `--prune`) because landmark extraction fails
  quietly — MediaPipe returns something for every frame, so a corpus where hands
  were never found trains without error.
- **Push-to-sign webcam demo** and a self-recording tool.
- **Dataset decision:** AUTSL (43 signers, official 31/6/6 split) as the primary
  target over BosphorusSign22k; open English data (WLASL) rejected for killing
  the TİD differentiator and lacking an official signer-independent split.
- **First results (LSA64, plumbing test — Argentinian, gloved signers, not TİD):**
  signer-independent over 64 signs, Transformer 97.1% / BiLSTM 88.0% top-1.
  Augmentation was worth 4.9 points at 26 glosses and nothing at 64; the
  Transformer's lead over the BiLSTM widened with the vocabulary. Both measured,
  both against the plan's assumptions.
- Fixes found by running it: MediaPipe cannot load its graph from a non-ASCII
  Windows path (the project directory was renamed to `signa`); torch threading
  hurts a model this small (475 ms/batch at one thread, 3900 at four);
  `resample` was calling `np.interp` 152 times per clip.
