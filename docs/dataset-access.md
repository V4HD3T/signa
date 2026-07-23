# Getting the data

Three datasets, in the order you should pursue them. The point of the ordering
is that nothing is ever blocked: something is downloadable today, the main
dataset needs a registration rather than a negotiation, and the slowest option
runs in the background costing nothing.

| | Language | Signs | Signers | Clips | Gate |
| --- | --- | --- | --- | --- | --- |
| **AUTSL** — primary | TİD | 226 | 43 | 38,336 | CodaLab registration |
| **LSA64** — plumbing test | LSA (Argentinian) | 64 | 10 | 3,200 | direct download |
| **BosphorusSign22k** — second dataset | TİD | 744 | 6 | 22,542 | research EULA by email |

## 1. AUTSL — the one to build on

Ankara University Turkish Sign Language dataset. 226 signs, 43 signers, 38,336
samples, Kinect v2 RGB + depth + skeleton at 512×512, recorded indoors and out
with varying backgrounds and lighting.

Three things make it the better primary source:

**43 signers.** BosphorusSign22k has six. Signer-independent generalisation is
the entire difficulty of this problem, and it is learned from signer diversity —
31 training signers teach a model what is invariant across people in a way that
four cannot. At ~170 samples per sign versus ~30, augmentation stops being the
thing holding the result together.

**The split is already defined, and it is signer-independent.** 31 signers train
/ 6 validation / 6 test — 28,142 / 4,418 / 3,742 samples. That is exactly the
three-way protocol `train.py` implements, so there is nothing to invent and
nothing to argue about in the defence. Pass it explicitly:

```bash
python -m signa.train --test-signers <the 6> --val-signers <the 6>
```

**A published leaderboard.** It was the CVPR 2021 ChaLearn "Large Scale Signer
Independent Isolated SLR" challenge — 132 teams in the RGB track, winners above
96%. That number is a multi-modal ensemble (RGB + depth + pose + optical flow),
not a landmark BiLSTM, so do not treat it as your target. Treat it as the thing
you position against: *"the challenge winner reaches 96% with a four-stream
ensemble over RGB and depth; this reaches X% from hand and pose landmarks alone,
running live on a webcam at 30 fps."* That is a better story than a bigger
number, because it is the honest one and it explains the gap.

### Access

Registration on CodaLab, not an email thread: the archives are encrypted and
**the decryption keys are provided on CodaLab after registration**. Start at

- <https://cvml.ankara.edu.tr/datasets/>
- → <http://chalearnlap.cvc.uab.es/dataset/40/description/>

A sample archive needs no registration at all, and is enough to check the
extraction path before committing to a 38k-clip download:
<https://cvml.ankara.edu.tr/autsl/examples.zip>

> **Verify this first.** The challenge ran in 2021 and key release was described
> as following "the schedule of the challenge". Whether keys are still handed out
> on a closed competition is the one thing to confirm before planning around it —
> do it today, it takes minutes. If the keys are gone, mail the organisers; that
> is the fallback, not the plan.

Academic and research use only; commercial use is not permitted.

## 2. LSA64 — run the pipeline today

3,200 clips, 64 signs, 10 signers, direct download, no registration, no forms:
<https://facundoq.github.io/datasets/lsa64/>

This is not a result you will report. It is Argentinian Sign Language, and the
signers wore **coloured gloves** to simplify hand segmentation — MediaPipe's hand
model is trained on bare hands, so detection quality on gloved hands is an open
question and quite possibly poor. Check the detection rate (the presence flags in
each extracted clip) before drawing any conclusion from an accuracy number here.

What it is genuinely good for: real multi-signer video, with a real
signer-independent split, available in the next ten minutes. Extract it, train
on it, run the demo. Every seam in this repo gets exercised on real data while
AUTSL is still downloading — and if MediaPipe struggles with the gloves, that
itself is the kind of finding worth a paragraph in the report.

## 3. BosphorusSign22k — leave it running in the background

744 glosses across health, finance and daily life, six signers, 22,542 clips,
1080p. Access is per-researcher against a signed EULA, by email:

| | |
| --- | --- |
| Oğulcan Özdemir | ogulcan.ozdemir@boun.edu.tr |
| Alp Kındıroğlu | alp.kindiroglu@boun.edu.tr |

(Boğaziçi University, Perceptual Intelligence Laboratory. Verify both addresses
against the current project page before sending — lab contact details drift.)

Send it anyway, even with AUTSL as the primary. It costs one email and an
unbounded but *parallel* wait, and having both unlocks the single most
interesting experiment available here: **train on AUTSL, test on
BosphorusSign22k**. Cross-dataset generalisation on overlapping glosses — two
independent recording setups, different signers, different cameras — is a far
stronger portfolio result than one more point of accuracy on a benchmark, and
almost nobody at this level does it.

### Draft request

> **Konu:** BosphorusSign22k veri seti erişim talebi — [Üniversite], lisans bitirme projesi
>
> Sayın Hocam,
>
> Ben [Üniversite] [Bölüm] son sınıf öğrencisi Vahdet Eren Bozyil. Lisans
> bitirme projem kapsamında Türk İşaret Dili için izole kelime tanıma üzerine
> çalışıyorum: MediaPipe ile çıkarılan el ve üst gövde landmark'ları üzerinde
> BiLSTM tabanlı bir sınıflandırıcı eğitip, webcam üzerinden çalışan bir demo
> hazırlamayı hedefliyorum.
>
> Çalışmamın birincil veri kümesi olarak AUTSL'i kullanıyorum. BosphorusSign22k'ya
> erişebilmem durumunda, iki veri kümesinde ortak olan glosslar üzerinde
> veri-kümeleri-arası genelleme (bir kümede eğitip diğerinde test etme) deneyi
> yapmayı planlıyorum; farklı kayıt ortamı ve farklı işaretleyicilerle elde
> edilecek bu sonucun tek bir kümedeki doğruluk oranından daha bilgilendirici
> olacağını düşünüyorum.
>
> Veri setine araştırma amaçlı erişim için imzalamam gereken EULA'yı ve
> izlemem gereken adımları tarafıma iletebilir misiniz? Veriyi yalnızca bu
> akademik çalışma kapsamında kullanacağımı, üçüncü kişilerle
> paylaşmayacağımı ve yayın/sunum durumunda ilgili makalelerinize atıf
> yapacağımı taahhüt ederim.
>
> İlginiz için şimdiden teşekkür eder, iyi çalışmalar dilerim.
>
> Saygılarımla,
> Vahdet Eren Bozyil
> [E-posta] · [Öğrenci no] · [Danışman adı]

## Why not WLASL / MS-ASL

The obvious pivot when a dataset is gated is to switch to open ASL data. It is
the wrong trade here, for four reasons:

1. **It costs the project its differentiator.** A TİD recogniser is a small
   field with real local impact. An ASL word classifier is a crowded one, where
   the comparison set is papers with far more compute than a graduation project.
2. **The numbers get worse, not better.** WLASL is web-scraped and in the wild —
   varying resolution, framerate and background. Published *state-of-the-art*
   pose-based results are around 80.9% top-1 on WLASL100 and 64.2% on WLASL300,
   and those are tuned transformers. A BiLSTM baseline lands well below that.
3. **No official signer-independent split.** That is the most defensible thing
   about this project's protocol, and WLASL does not give it to you.
4. **Known annotation problems** — homographs sharing a gloss, typos in labels —
   which some researchers avoid the original release over.

If a second *language* is ever wanted, it is a stronger claim as an extension
("the same landmark pipeline transfers to ASL") than as a substitute.

## Before extracting anything

```bash
python -m signa.extract --videos data/raw --dry-run
```

It prints the `(gloss, signer)` pairs it parsed without touching MediaPipe.
Check them. A wrong filename convention does not crash — it produces a split
where the same signer sits on both sides, and an accuracy number that means
nothing.

## A note on the provided skeletons

AUTSL ships Kinect skeletons (25 joints); BosphorusSign22k ships Kinect and
OpenPose. **Do not train on either.** The demo has to run MediaPipe — OpenPose
is not web-native and far too heavy for a live webcam loop — and the joint
order, count and normalisation do not match. Training on one skeleton and
inferring with another is a silent train/inference mismatch: the model
converges, the demo loads, the predictions are noise.

Extract your own MediaPipe landmarks from the RGB video. Use the provided
skeletons only as a sanity check — if MediaPipe fails to find hands on clips
where Kinect tracked them, that is worth knowing before you blame the model.
