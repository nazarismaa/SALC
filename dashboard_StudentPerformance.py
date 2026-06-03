import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(page_title="SALC Performance Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data_StudentsPerformance_Processed.csv")
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Filter Analisis")
lunch_filter = st.sidebar.multiselect("Pilih Lunch (Proksi Status Ekonomi):", options=df['lunch'].unique(), default=df['lunch'].unique())
prep_filter = st.sidebar.multiselect("Pilih Status Kursus:", options=df['test preparation course'].unique(), default=df['test preparation course'].unique())

filtered_df = df[(df['lunch'].isin(lunch_filter)) & (df['test preparation course'].isin(prep_filter))]

# --- MAIN PAGE ---
st.title("🎓 SALC: Performance & Risk Analysis Dashboard")

# Baris 1: Metrik Utama
col1, col2, col3 = st.columns(3)
col1.metric("Total Siswa", len(filtered_df))
col2.metric("Rata-rata Skor", round(filtered_df['average_score'].mean(), 2))
col3.metric("Persentase Risiko Tinggi", f"{round((filtered_df['is_at_risk'].mean())*100, 1)}%")

# Baris 2: Analisis Pertanyaan Bisnis 1
st.header("1. Pengaruh Faktor Latar Belakang terhadap Performa")
fig1, ax1 = plt.subplots(figsize=(10, 4))
pivot_data = filtered_df.pivot_table(values='average_score', index='parental level of education', 
                                    columns='test preparation course', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax1)
st.pyplot(fig1)

# Baris 3: Analisis Pertanyaan Bisnis 2
st.header("2. Deteksi Dini: Karakteristik Siswa Berisiko")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Risiko per Tipe Lunch (Proksi Status Ekonomi)")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=filtered_df, x='lunch', y='is_at_risk', ax=ax2, palette='magma')
    st.pyplot(fig2)
with col_b:
    st.subheader("Distribusi Skor Matematika")
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered_df['math score'], kde=True, color='salmon', ax=ax3)
    st.pyplot(fig3)