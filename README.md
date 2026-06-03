---
title: SALC AI Service
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# SALC AI Service

Backend AI/ML untuk Sistem Adaptive Learning Companion (SALC).
Menyediakan 3 model untuk early-warning siswa:

1. **NLP Feedback** — analisis jawaban siswa (Keras LSTM)
2. **Activity Risk** — prediksi risiko dari pola belajar (Random Forest)
3. **Performance Risk** — prediksi risiko dari nilai akademik (Gradient Boosting)

## Endpoint

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET    | `/` | Health check |
| POST   | `/predict` | NLP feedback dari teks jawaban |
| POST   | `/api/predict/activity` | Risk dari aktivitas belajar |
| POST   | `/api/predict/performance` | Risk dari nilai akademik |

Dokumentasi interaktif: `/docs`

## Struktur

```
.
├── api/main.py          # FastAPI endpoints
├── utils/
│   ├── inference.py     # NLP inference
│   └── preprocessor.py  # text cleaning
├── model/               # file model (.pkl, .keras)
├── Dockerfile
└── requirements.txt
```

## Run Lokal

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 7860
```

Akses: http://localhost:7860/docs
