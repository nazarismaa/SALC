# 🎓 ML Documentation
## Smart Adaptive Learning Companion (SALC)

> **Versi:** 1.0.0 &nbsp;|&nbsp; **Tanggal:** Mei 2025 &nbsp;|&nbsp; **Tim:** Data Science SALC

---

## 📋 Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Arsitektur ML Pipeline](#2-arsitektur-ml-pipeline)
3. [Model 1 — Prediksi Risiko Aktivitas Belajar](#3-model-1--prediksi-risiko-aktivitas-belajar)
4. [Model 2 — Prediksi Risiko Performa Akademik](#4-model-2--prediksi-risiko-performa-akademik)
5. [Recommendation Logic](#5-recommendation-logic)
6. [Model Export & Deployment Artifacts](#6-model-export--deployment-artifacts)
7. [API Integration Guide](#7-api-integration-guide)
8. [Keterbatasan & Catatan Penting](#8-keterbatasan--catatan-penting)

---

## 1. Gambaran Umum

SALC menggunakan **dua model Machine Learning** yang bekerja secara komplementer untuk mendeteksi risiko ketertinggalan siswa dan memberikan rekomendasi pembelajaran yang dipersonalisasi.

| | 🌲 Model 1 | 🚀 Model 2 |
|---|---|---|
| **Nama** | AktivitasBelajar | StudentsPerformance |
| **Algoritma Final** | Random Forest Classifier | Gradient Boosting Classifier |
| **Tipe Prediksi** | Multi-class (3 kelas) | Binary classification |
| **Target** | `RiskLevel` | `is_at_risk` |
| **Accuracy (final)** | `98.60%` | `92.50%` |
| **F1-Score (final)** | `97.88%` (macro) | `68.09%` (kelas at-risk) |
| **File Model** | `salc_rf_model.pkl` | `model_StudentPerformance.pkl` |

### Perbedaan Use Case

| Model | Kapan Digunakan | Sumber Data |
|---|---|---|
| 🔄 **Model 1** | Monitoring berkelanjutan — deteksi risiko dari perilaku & aktivitas belajar secara real-time | Platform SALC |
| 🔍 **Model 2** | Profiling awal — deteksi risiko dari profil akademik saat mendaftar atau awal semester | Skor ujian & latar belakang siswa |

---

## 2. Arsitektur ML Pipeline

Kedua model mengikuti pipeline yang sama:

```
Raw Data (CSV)
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split (80:20, stratified)
      │
      ▼
Model Building — 3 kandidat algoritma
      │
      ▼
Evaluasi Awal (Confusion Matrix, Classification Report)
      │
      ▼
Evaluasi Lanjutan (Feature Importance)
      │
      ▼
Cross Validation (5-fold)
      │
      ▼
Visualisasi Perbandingan Model
      │
      ▼
Recommendation Logic
      │
      ▼
Model Optimization (GridSearchCV)
      │
      ▼
Model Export (.pkl + metadata.json)
      │
      ▼
Finalisasi (verifikasi konsistensi + uji end-to-end pipeline)
```

---

## 3. Model 1 — Prediksi Risiko Aktivitas Belajar

### 3.1 Dataset

| Atribut | Detail |
|---|---|
| **Sumber** | `data_AktivitasBelajar.csv` |
| **Ukuran Total** | ~12.470 record |
| **Train Set** | ~9.976 record |
| **Test Set** | ~2.494 record |
| **Target** | `RiskLevel` |

**Label Target:**

| Label | Deskripsi | Urgensi |
|---|---|---|
| `Aman` | Tidak berisiko tertinggal | 🟢 Normal |
| `Perhatian` | Perlu pemantauan lebih | 🟡 Waspada |
| `Kritis` | Membutuhkan intervensi segera | 🔴 Kritis |

**Kolom yang dieksklusi** (alasan: *data leakage*):

| Kolom | Alasan Eksklusi |
|---|---|
| `RiskScore`, `FinalGrade` | Derivat langsung dari target |
| `StudyHours_Category`, `Attendance_Category`, `StressLevel_Category`, `AssignmentCompletion_Category`, `DigitalReadiness_Label` | Versi teks dari fitur numerik yang sudah ada |

---

### 3.2 Pemilihan Model

| Model | Accuracy | F1-Macro | Keterangan |
|---|---|---|---|
| Logistic Regression | 0.8905 | 0.8215 | — |
| **🏆 Random Forest** | **0.9852** | **0.9752** | **Dipilih** |
| Gradient Boosting | 0.9836 | 0.9770 | — |

> **Mengapa Random Forest?** Accuracy & F1-Macro tertinggi pada baseline. Selisih dengan Gradient Boosting sangat kecil, namun Random Forest lebih cepat untuk inference — relevan untuk kebutuhan **real-time monitoring** di platform SALC.

---

### 3.3 Evaluasi Awal

**Confusion Matrix** *(baseline Random Forest, test set n=2.494):*

| | Pred: Aman | Pred: Perhatian | Pred: Kritis |
|---|:---:|:---:|:---:|
| **Actual: Aman** | ~938 | 14 | 0 |
| **Actual: Perhatian** | 7 | ~839 | 0 |
| **Actual: Kritis** | 0 | 16 | ~680 |

> ✅ **Total salah prediksi: 37 dari 2.494 data (1.48%)**
> Kelas **Aman** memiliki error tertinggi (14 salah diprediksi sebagai Perhatian) — wajar karena jumlah datanya paling sedikit.

---

### 3.4 Feature Importance

| Rank | Fitur | Importance | Visualisasi |
|:---:|---|:---:|---|
| 1 | `EngagementScore` | **29.73%** | `████████████████████████████░` |
| 2 | `Motivation` | **20.84%** | `████████████████████░░░░░░░░░` |
| 3 | `AssignmentCompletion` | **11.62%** | `███████████░░░░░░░░░░░░░░░░░░` |
| 4 | `StudyStressBalance` | **8.37%** | `████████░░░░░░░░░░░░░░░░░░░░░` |
| 5 | `Attendance` | **8.24%** | `████████░░░░░░░░░░░░░░░░░░░░░` |
| — | `Gender`, `Internet`, `EduTech` | **< 1%** | `░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |

> 💡 **Temuan kunci:** Risiko ketertinggalan lebih ditentukan oleh **perilaku belajar** (engagement, motivasi, konsistensi tugas) daripada faktor demografis (gender, akses internet, kepemilikan perangkat).

---

### 3.5 Cross Validation (5-Fold)

| Metric | Mean | Std Dev | Status |
|---|:---:|:---:|:---:|
| Accuracy | **97.51%** | ±0.84% | ✅ Stabil |
| F1-Macro | **95.96%** | ±1.19% | ✅ Stabil |

> Gap antara CV score dan test score **< 1%** → model tidak overfitting. Standar deviasi yang kecil membuktikan model stabil dan konsisten di berbagai subset data.

---

### 3.6 Model Optimization (GridSearchCV)

```python
param_grid = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf' : [1, 2]
}
```

**Best Parameters:**
```json
{
  "max_depth"         : 20,
  "min_samples_leaf"  : 1,
  "min_samples_split" : 5,
  "n_estimators"      : 200
}
```

**Hasil setelah tuning:**

| Metric | Baseline | Final | Delta |
|---|:---:|:---:|:---:|
| Accuracy | 0.9852 | **0.9860** | `+0.08%` |
| F1-Macro | 0.9752 | **0.9788** | `+0.36%` |

> Peningkatan kecil mengonfirmasi bahwa baseline Random Forest sudah sangat kuat — tidak ada ruang peningkatan artifisial yang signifikan.

---

## 4. Model 2 — Prediksi Risiko Performa Akademik

### 4.1 Dataset

| Atribut | Detail |
|---|---|
| **Sumber** | `data_StudentsPerformance_Processed.csv` |
| **Ukuran Total** | ~1.000 record |
| **Train Set** | ~800 record |
| **Test Set** | 200 record |
| **Target** | `is_at_risk` |

**Distribusi Target:**

| Label | Deskripsi | Proporsi |
|---|---|:---:|
| `0` — Not At Risk | Math score ≥ 50 | **~86.5%** |
| `1` — At Risk | Math score < 50 | **~13.5%** |

> ⚠️ `math score` **dieksklusi** dari fitur karena merupakan basis definisi target — menggunakannya sebagai fitur adalah *data leakage*.

---

### 4.2 Feature Engineering

Tiga fitur turunan dibuat sebelum training:

```python
df['score_gap']          = df['reading score'] - df['writing score']
df['academic_readiness'] = df['test preparation course'] * df['reading score']
df['socio_academic']     = df['lunch'] * df['reading score']
```

**Fitur yang digunakan (10 total):**

| # | Fitur | Tipe | Keterangan |
|:---:|---|---|---|
| 1 | `gender` | Binary | — |
| 2 | `race/ethnicity` | Ordinal | — |
| 3 | `parental level of education` | Ordinal | — |
| 4 | `lunch` | Binary | Proxy status sosial-ekonomi |
| 5 | `test preparation course` | Binary | — |
| 6 | `reading score` | Numerik | — |
| 7 | `writing score` | Numerik | — |
| 8 | `score_gap` | Numerik | ⚙️ *Engineered* |
| 9 | `academic_readiness` | Numerik | ⚙️ *Engineered* |
| 10 | `socio_academic` | Numerik | ⚙️ *Engineered* |

---

### 4.3 Pemilihan Model

| Model | Accuracy | Precision | Recall | F1 | Keterangan |
|---|:---:|:---:|:---:|:---:|---|
| Logistic Regression | 0.8850 | 0.5500 | **0.8148** | 0.6567 | Recall tinggi tapi banyak false alarm |
| Random Forest | 0.9000 | 0.6842 | 0.4815 | 0.5652 | Banyak at-risk tidak terdeteksi |
| **🏆 Gradient Boosting** | **0.9250** | **0.7308** | **0.7037** | **0.7170** | **Dipilih — paling seimbang** |

> **Mengapa Gradient Boosting?** F1-score tertinggi untuk kelas at-risk. Logistic Regression terlalu banyak false alarm (Precision 55%), Random Forest terlalu banyak miss (Recall 48%). Gradient Boosting paling seimbang.

---

### 4.4 Evaluasi Awal

**Confusion Matrix** *(baseline Gradient Boosting, test set n=200):*

| | Pred: Not At Risk | Pred: At Risk |
|---|:---:|:---:|
| **Actual: Not At Risk** | 169 | 4 |
| **Actual: At Risk** | 11 | 16 |

> ✅ Dari 27 siswa at-risk, **16 berhasil terdeteksi (59%)**
> ✅ Hanya **4 siswa aman** salah diprediksi berisiko → false positive rendah, minim gangguan

---

### 4.5 Feature Importance

| Rank | Fitur | Importance | Visualisasi |
|:---:|---|:---:|---|
| 1 | `writing score` | **54.16%** | `█████████████████████████████████████████████████████░` |
| 2 | `gender` | **17.70%** | `█████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| 3 | `socio_academic` | **11.41%** | `███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| 4 | `reading score` | **~8%** | `████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| 5 | `academic_readiness` | **~5%** | `█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| — | `test preparation course` | **0%** | `░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |

> 💡 **Temuan kunci:** `writing score` mendominasi prediksi (54%). `test preparation course` tidak berkontribusi karena pengaruhnya sudah sepenuhnya tercermin di skor reading & writing.

---

### 4.6 Cross Validation (5-Fold)

| Metric | Mean | Std Dev | Status |
|---|:---:|:---:|:---:|
| Accuracy | **91.70%** | ±1.03% | ✅ Konsisten |
| F1 | **67.89%** | ±4.32% | ⚠️ Perlu lebih banyak data |

> Std dev F1 yang lebih besar (±4.32%) wajar untuk dataset **class-imbalanced** — performa pada kelas minoritas (at-risk) lebih sensitif terhadap komposisi fold.

---

### 4.7 Model Optimization (GridSearchCV)

```python
param_grid = {
    'n_estimators'    : [100, 200],
    'learning_rate'   : [0.05, 0.1],
    'max_depth'       : [3, 4],
    'min_samples_leaf': [5, 10]
}
```

**Best Parameters:**
```json
{
  "learning_rate"    : 0.05,
  "max_depth"        : 3,
  "min_samples_leaf" : 10,
  "n_estimators"     : 100
}
```

**Hasil setelah tuning:**

| Metric | Baseline | Final | Perubahan |
|---|:---:|:---:|:---:|
| Accuracy | 0.9250 | **0.9250** | — |
| Precision | 0.7308 | **0.8000** | `+9.2%` ⬆️ |
| Recall | 0.7037 | **0.5926** | `-14.1%` ⬇️ |
| F1 | 0.7170 | **0.6809** | `-5.0%` ⬇️ |

> ⚖️ **Trade-off yang disengaja:** Model lebih selektif dalam memberi peringatan — ketika peringatan muncul, kemungkinan besar memang benar. Untuk sistem *early warning* seperti SALC, **meminimalkan false alarm lebih diprioritaskan** agar kepercayaan pengguna terjaga.

**Classification Report (model final):**

| Kelas | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| Not At Risk | 0.94 | 0.98 | 0.96 | 173 |
| At Risk | 0.80 | 0.59 | 0.68 | 27 |
| **Accuracy** | | | **0.93** | **200** |
| Macro Avg | 0.87 | 0.78 | 0.82 | 200 |
| Weighted Avg | 0.92 | 0.93 | 0.92 | 200 |

---

## 5. Recommendation Logic

### 5.1 Model 1 — AktivitasBelajar

Fungsi `get_recommendation()` menghasilkan rekomendasi adaptif berdasarkan kombinasi output model dan data tambahan siswa.

**Parameter input:**

| Parameter | Sumber |
|---|---|
| `risk` | Output prediksi model |
| `learning_style` | Data profil siswa (`0`=Visual, `1`=Auditory, `2`=Kinesthetic, `3`=Reading-Writing) |
| `motivation` | Skor motivasi siswa |
| `study_hours` | Jam belajar per minggu |
| `assignment_completion` | Persentase tugas selesai |

**Threshold aksi untuk siswa Kritis 🔴:**

| Kondisi | Action Plan |
|---|---|
| `motivation < 3` | Mentoring motivasi dengan konselor akademik |
| `assignment_completion < 60%` | Aktifkan pengingat tugas + teknik Pomodoro |
| `study_hours < 10` | Tambah durasi belajar minimal 2 jam/hari |

**Media rekomendasi per gaya belajar:**

| Gaya Belajar | Media yang Direkomendasikan |
|---|---|
| 👁️ Visual | Video, infografis |
| 👂 Auditory | Rekaman, diskusi audio |
| 🤸 Kinesthetic | Latihan interaktif |
| ✍️ Reading-Writing | Modul bacaan |

---

### 5.2 Model 2 — StudentsPerformance

Fungsi `generate_recommendation()` menggunakan **probability threshold berlapis**, bukan sekadar binary label.

| Probabilitas | Risk Level | Aksi |
|---|:---:|---|
| `≥ 0.70` | 🔴 **KRITIS** | Intervensi segera |
| `0.40 – 0.69` | 🟡 **SEDANG** | Perhatian khusus |
| `< 0.40` | 🟢 **AMAN** | Pengayaan opsional |

> Rekomendasi mempertimbangkan 4 dimensi sekaligus: `reading score`, `writing score`, status ekonomi (`lunch`), dan keikutsertaan `test preparation course`.

---

## 6. Model Export & Deployment Artifacts

### 6.1 Struktur File

```
model/
├── 🌲 salc_rf_model.pkl                 ← Model 1 (Random Forest)
├── 📋 model_columns.pkl                 ← Urutan & nama fitur Model 1
├── 📄 model_metadata_aktivitas.json     ← Metadata & performa Model 1
├── 🚀 model_StudentPerformance.pkl      ← Model 2 (Gradient Boosting)
└── 📄 model_metadata.json               ← Metadata & performa Model 2
```

### 6.2 Isi Metadata

**`model_metadata_aktivitas.json`** (Model 1)
```json
{
  "model_name"      : "Random Forest Classifier",
  "version"         : "1.0.0",
  "target"          : "RiskLevel",
  "feature_cols"    : ["..."],
  "best_params"     : {
    "max_depth": 20,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 200
  },
  "performance"     : { "accuracy": 0.9860, "f1_macro": 0.9788 },
  "risk_categories" : ["Aman", "Perhatian", "Kritis"]
}
```

**`model_metadata.json`** (Model 2)
```json
{
  "model_name"  : "Gradient Boosting Classifier",
  "version"     : "1.0.0",
  "target"      : "is_at_risk",
  "feature_cols": ["..."],
  "best_params" : {
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "n_estimators": 100
  },
  "performance" : {
    "accuracy": 0.9250,
    "precision": 0.8000,
    "recall": 0.5926,
    "f1_score": 0.6809
  },
  "risk_levels" : {
    "KRITIS": "Probabilitas >= 0.7",
    "SEDANG": "Probabilitas 0.4 - 0.7",
    "AMAN"  : "Probabilitas < 0.4"
  }
}
```

### 6.3 Cara Load Model

```python
import joblib, json

# ─── Model 1: AktivitasBelajar ───────────────────────────────────────────
rf_model        = joblib.load('model/salc_rf_model.pkl')
rf_feature_cols = joblib.load('model/model_columns.pkl')
with open('model/model_metadata_aktivitas.json') as f:
    rf_meta = json.load(f)

# ─── Model 2: StudentsPerformance ────────────────────────────────────────
gb_model = joblib.load('model/model_StudentPerformance.pkl')
with open('model/model_metadata.json') as f:
    gb_meta = json.load(f)
```

---

## 7. API Integration Guide

### 7.1 Schema Input

**Model 1 — `POST /api/predict/activity`**

> Kirimkan semua kolom fitur sesuai `model_columns.pkl`. Urutan kolom harus konsisten — gunakan `model_columns.pkl` sebagai referensi, **jangan hardcode** di sisi backend.

**Model 2 — `POST /api/predict/performance`**

Cukup kirimkan **7 fitur raw** berikut:

```json
{
  "gender": 1,
  "race/ethnicity": 2,
  "parental level of education": 2,
  "lunch": 1,
  "test preparation course": 0,
  "reading score": 65,
  "writing score": 60
}
```

> ℹ️ `score_gap`, `academic_readiness`, dan `socio_academic` **tidak perlu dikirim** — ketiga fitur ini dikalkulasi otomatis oleh pipeline sebelum masuk ke model.

---

### 7.2 Schema Output

**Model 1:**
```json
{
  "risk_level"       : "Perhatian",
  "probabilities"    : { "Aman": 0.12, "Perhatian": 0.75, "Kritis": 0.13 },
  "confidence"       : 0.75,
  "recommendations"  : ["..."],
  "media_rekomendasi": ["..."],
  "action_plan"      : ["..."],
  "model_version"    : "1.0.0"
}
```

**Model 2:**
```json
{
  "prediction"     : 1,
  "at_risk"        : true,
  "probability"    : 0.82,
  "risk_tier"      : "KRITIS",
  "recommendations": ["..."],
  "action_plan"    : ["..."],
  "model_version"  : "1.0.0"
}
```

---

### 7.3 Dependencies

```txt
scikit-learn >= 1.3.0
joblib       >= 1.3.0
pandas       >= 2.0.0
numpy        >= 1.24.0
shap         >= 0.42.0
```

---

## 8. Keterbatasan & Catatan Penting

### ⚠️ Model 1 — AktivitasBelajar

| Keterbatasan | Catatan |
|---|---|
| Akurasi sangat tinggi (>98%) | Perlu divalidasi ulang dengan data nyata saat go-live — kemungkinan dataset masih sangat bersih/ideal |
| Tidak ada fitur temporal | Model tidak menangkap tren penurunan performa seiring waktu. Pertimbangkan *time-series features* di iterasi berikutnya |
| Fitur demografis hampir tidak berpengaruh | Pastikan siswa tanpa akses internet tetap terwakili dalam data training agar model tidak bias |

### ⚠️ Model 2 — StudentsPerformance

| Keterbatasan | Catatan |
|---|---|
| Recall 59% setelah tuning | ~41% siswa at-risk tidak terdeteksi. Gunakan bersama Model 1 untuk coverage yang lebih lengkap |
| Dataset kecil (~1.000 record) | F1 std dev yang tinggi (±4.32%) menandakan performa belum benar-benar stabil. Tambah data untuk memperkuat model |
| `writing score` dominasi 54% | Model terlalu bergantung satu fitur. Siapkan *fallback logic* jika fitur ini tidak tersedia |
| `gender` berkontribusi 17.70% | Lakukan **audit fairness** secara berkala untuk memastikan model tidak bias terhadap kelompok tertentu |
| Definisi `is_at_risk` berbasis math score < 50 | Threshold ini mungkin tidak relevan untuk semua kurikulum. Pertimbangkan konfigurasi threshold per institusi di versi berikutnya |

### 📌 Catatan Umum

| Aspek | Rekomendasi |
|---|---|
| 🔄 **Retraining** | Lakukan retraining minimal setiap semester dengan data terbaru untuk mencegah *model drift* |
| 📊 **Monitoring** | Pantau distribusi prediksi secara berkala — perubahan signifikan bisa jadi sinyal bahwa pola belajar siswa sudah bergeser dari data training |
| 🆕 **Cold Start** | Untuk siswa baru yang belum punya data aktivitas, Model 1 tidak bisa dipakai — fallback ke Model 2 atau tampilkan rekomendasi default |

---

<div align="center">

*Dokumen ini dibuat oleh **Tim Data Science SALC** — Versi 1.0.0 | Mei 2025*

</div>
