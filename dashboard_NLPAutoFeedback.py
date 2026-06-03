import streamlit as st
import pandas as pd
import numpy as np
# Import library NLP kamu di sini, contoh:
# from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="SALC Auto Feedback", layout="centered")

st.title("🎓 SALC: NLP Auto Feedback")
st.write("Masukkan jawaban siswa untuk mendapatkan umpan balik instan.")

# --- BAGIAN LOGIKA NLP ---
def get_auto_feedback(student_answer):
    # Di sini kamu masukkan logika dari Notebook kamu
    # Contoh sederhana berbasis kata kunci:
    keywords = ["penting", "belajar", "inovasi", "teknologi"]
    score = 0
    for word in keywords:
        if word in student_answer.lower():
            score += 25
            
    if score >= 75:
        feedback = "Luar biasa! Jawabanmu sangat komprehensif."
    elif score >= 25:
        feedback = "Jawaban cukup baik, namun perlu diperdalam pada poin tertentu."
    else:
        feedback = "Jawaban belum sesuai. Silakan tinjau kembali materi terkait."
        
    return score, feedback

# --- BAGIAN TAMPILAN STREAMLIT ---
student_answer = st.text_area("Tulis jawaban esai di sini:", height=150)

if st.button("Analisis Jawaban"):
    if student_answer.strip():
        with st.spinner("Sedang menganalisis jawaban..."):
            score, feedback = get_auto_feedback(student_answer)
            
            st.subheader("Hasil Analisis:")
            st.metric("Skor Otomatis", f"{score}/100")
            st.success(feedback)
    else:
        st.warning("Jawaban masih kosong!")

# Tambahan: Preview Dataset (opsional)
with st.expander("Lihat Dataset Referensi"):
    st.write("Dataset ini digunakan sebagai pembanding otomatis:")
    # st.write(pd.read_csv("data_NLPAutoFeedback.csv")) # Pastikan file ada