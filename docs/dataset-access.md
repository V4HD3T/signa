# Getting the dataset

BosphorusSign22k is not a public download. Access is granted per researcher
against a signed EULA, so **this is the long pole of the whole project** — send
the request before writing another line of code, and build against self-recorded
clips while it is pending.

## Who to write to

| | |
| --- | --- |
| Oğulcan Özdemir | ogulcan.ozdemir@boun.edu.tr |
| Alp Kındıroğlu | alp.kindiroglu@boun.edu.tr |

(Boğaziçi University, Perceptual Intelligence Laboratory.) Verify both addresses
against the current project page before sending — lab contact details drift, and
a bounced mail costs a week.

## Draft request

Copy, fill the bracketed fields, send from your university address if you have
one — it materially changes how a research-use request reads.

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
> Bu çalışma için BosphorusSign22k veri setini kullanmak istiyorum. Özellikle
> 174 glossluk genel/günlük alt kümesini ve yayınlarınızda tanımlanan
> signer-independent bölünmeyi temel almayı planlıyorum; böylece sonuçlarımı
> literatürdeki baseline'larla karşılaştırılabilir şekilde raporlayabilirim.
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

## While you wait

Nothing below needs the dataset:

```bash
python -m signa.record --gloss merhaba --signer me --count 5
```

Record 4–5 glosses, extract, train a throwaway model, run the demo. It proves
every seam fits — filename parsing, the frame layout, normalisation,
train/inference consistency — so that when the archive lands the only new
variable is the data itself.

## When the archive arrives

The authors ship pre-extracted skeletons (Kinect v2, 25 3D joints; OpenPose,
25 body + 70 face + 2×21 hand, 2D) alongside the raw 1080p video. **Do not train
on those.** The demo has to run MediaPipe — OpenPose is not web-native and is far
too heavy for a live webcam loop — and MediaPipe's joint order, count and
normalisation do not match either format. Training on one skeleton and inferring
with another is a silent train/inference mismatch: the model converges, the demo
loads, and the predictions are noise.

Extract your own MediaPipe landmarks from the raw video. Use the provided
skeletons only as a sanity check — if MediaPipe fails to find hands on clips
where OpenPose did, that is worth knowing before you blame the model.

First command to run, before extracting anything:

```bash
python -m signa.extract --videos data/raw --dry-run
```

It prints the `(gloss, signer)` pairs it parsed without touching MediaPipe.
Check them. A wrong filename convention does not crash — it produces a split
where the same signer sits on both sides, and an accuracy number that means
nothing.
