
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessor import clean_text

MAX_LEN = 500
SCORE_MIN = 1
SCORE_MAX = 6

MODEL_PATH = "model/salc_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

model = load_model(MODEL_PATH, compile=False)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

def kategori_to_feedback(kategori):
    feedbacks = {
        "baik": "Excellent answer! Your arguments are strong and clearly explained.",
        "cukup": "Good effort! Add more detail and supporting evidence.",
        "kurang": "This answer needs improvement. Try to explain your ideas more clearly."
    }
    return feedbacks.get(kategori)

def predict_feedback(text):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    pred_cat, pred_score = model.predict(padded, verbose=0)

    confidence = float(np.max(pred_cat[0]))

    score_raw = float(pred_score[0][0]) * (SCORE_MAX - SCORE_MIN) + SCORE_MIN

    score_raw = max(SCORE_MIN, min(SCORE_MAX, score_raw))

    score_100 = int(round((score_raw - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100))

    if score_raw >= 4.5:
        kategori = 'baik'
    elif score_raw >= 3:
        kategori = 'cukup'
    else:
        kategori = 'kurang'

    return {
        "kategori": kategori,
        "skor": score_100,
        "feedback": kategori_to_feedback(kategori),
        "confidence": round(confidence, 3)
    }
