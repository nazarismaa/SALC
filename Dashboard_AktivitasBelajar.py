import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set Konfigurasi Halaman
st.set_page_config(page_title="SALC Dashboard", layout="wide")

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data_AktivitasBelajar.csv")
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Filter Dashboard")
# Filter Gaya Belajar
learning_style_options = df['LearningStyle'].unique()
selected_style = st.sidebar.multiselect("Pilih Gaya Belajar:", 
                                        options=learning_style_options, 
                                        default=learning_style_options)

# Filter Data berdasarkan sidebar
filtered_df = df[df['LearningStyle'].isin(selected_style)]

# --- MAIN PAGE ---
st.title("Smart Adaptive Learning Companion (SALC) - Student Activity Analysis")
st.markdown("Dashboard ini menganalisis pola belajar siswa untuk memberikan rekomendasi adaptif.")

# --- BARIS 1: METRIK UTAMA ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Siswa", len(filtered_df))
with col2:
    st.metric("Rata-rata Jam Belajar", f"{round(filtered_df['StudyHours'].mean(), 1)} Jam")
with col3:
    st.metric("Penyelesaian Tugas", f"{round(filtered_df['AssignmentCompletion'].mean(), 1)}%")

st.divider()

# --- BARIS 2: PERTANYAAN BISNIS 1 (Segmentasi) ---
st.header("1. Segmentasi Pola Belajar & Tingkat Kelulusan")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Gaya Belajar vs Jam Belajar")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x='LearningStyle', y='StudyHours', hue='FinalGrade', palette='viridis', ax=ax)
    st.pyplot(fig)

with col_b:
    st.subheader("Pengaruh EduTech terhadap Kelulusan")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='EduTech', hue='FinalGrade', palette='magma', ax=ax)
    st.pyplot(fig)

# --- BARIS 3: PERTANYAAN BISNIS 2 (Prediktor Dini) ---
st.header("2. Deteksi Dini: Prediktor Penurunan Nilai")
st.markdown("Melihat tren bagaimana penyelesaian tugas mempengaruhi Grade akhir.")

# Line Plot Tren (Sesuai permintaan Anda sebelumnya)
fig_line, ax_line = plt.subplots(figsize=(10, 4))
trend_data = filtered_df.groupby('FinalGrade')['AssignmentCompletion'].mean().reset_index()
sns.lineplot(data=trend_data, x='FinalGrade', y='AssignmentCompletion', marker='o', color='teal', ax=ax_line)
ax_line.set_title("Tren Penurunan Penyelesaian Tugas terhadap Grade")
st.pyplot(fig_line)

# --- BARIS 4: REKOMENDASI ADAPTIF OTOMATIS ---
st.divider()
st.header("Rekomendasi Adaptif Sistem")
low_assignment = filtered_df[filtered_df['AssignmentCompletion'] < 70]
if not low_assignment.empty:
    st.warning(f"Perhatian! Ada {len(low_assignment)} siswa dengan penyelesaian tugas di bawah 70%. Dibutuhkan intervensi segera!")
else:
    st.success("Semua siswa dalam jalur belajar yang aman.")


# =============================================================
# BAGIAN BARU — Fitur dari Feature Engineering (EDA Notebook)
# Semua kolom berikut sudah tersedia di data_AktivitasBelajar.csv:
# StudyHours_Category, Attendance_Category, StressLevel_Category,
# AssignmentCompletion_Category, StudyEfficiency, StudyStressBalance,
# EngagementScore, DigitalReadiness_Label, RiskScore, RiskLevel
# =============================================================

# --- FILTER TAMBAHAN SIDEBAR (dari fitur FE) ---
st.sidebar.divider()
st.sidebar.subheader("Filter Fitur Baru")

# Filter Kategori Jam Belajar
if 'StudyHours_Category' in df.columns:
    study_cat_options = df['StudyHours_Category'].dropna().unique().tolist()
    selected_study_cat = st.sidebar.multiselect(
        "Kategori Jam Belajar:",
        options=study_cat_options,
        default=study_cat_options
    )
    filtered_df = filtered_df[filtered_df['StudyHours_Category'].isin(selected_study_cat)]

# Filter Risk Level
if 'RiskLevel' in df.columns:
    risk_options = df['RiskLevel'].dropna().unique().tolist()
    selected_risk = st.sidebar.multiselect(
        "Risk Level:",
        options=risk_options,
        default=risk_options
    )
    filtered_df = filtered_df[filtered_df['RiskLevel'].isin(selected_risk)]

# Filter Kesiapan Digital
if 'DigitalReadiness_Label' in df.columns:
    digital_options = df['DigitalReadiness_Label'].dropna().unique().tolist()
    selected_digital = st.sidebar.multiselect(
        "Kesiapan Digital:",
        options=digital_options,
        default=digital_options
    )
    filtered_df = filtered_df[filtered_df['DigitalReadiness_Label'].isin(selected_digital)]

st.divider()

# --- BARIS 5: EARLY WARNING SYSTEM (dari RiskScore & RiskLevel) ---
st.header("3. Early Warning System — Risk Score Terpadu")
st.markdown("Skor risiko gabungan dari 5 indikator: penyelesaian tugas (30%), kehadiran (25%), motivasi (20%), stres (15%), jam belajar (10%).")

if 'RiskLevel' in filtered_df.columns and 'RiskScore' in filtered_df.columns:

    # Metric cards per Risk Level
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    total = len(filtered_df)
    n_aman     = len(filtered_df[filtered_df['RiskLevel'] == 'Aman'])
    n_perhatian= len(filtered_df[filtered_df['RiskLevel'] == 'Perhatian'])
    n_kritis   = len(filtered_df[filtered_df['RiskLevel'] == 'Kritis'])

    with col_r1:
        st.metric("Total Siswa (Filter)", total)
    with col_r2:
        st.metric("Aman", f"{n_aman} siswa", f"{n_aman/total*100:.1f}%" if total > 0 else "0%")
    with col_r3:
        st.metric("Perhatian", f"{n_perhatian} siswa", f"{n_perhatian/total*100:.1f}%" if total > 0 else "0%")
    with col_r4:
        st.metric("Kritis", f"{n_kritis} siswa", f"{n_kritis/total*100:.1f}%" if total > 0 else "0%")

    col_ew1, col_ew2 = st.columns(2)

    with col_ew1:
        # Bar chart distribusi RiskLevel
        st.subheader("Distribusi Risk Level")
        risk_order  = ['Aman', 'Perhatian', 'Kritis']
        colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
        risk_counts = (
            filtered_df['RiskLevel']
            .value_counts()
            .reindex(risk_order)
            .fillna(0)
        )
        fig_risk, ax_risk = plt.subplots(figsize=(6, 4))
        bars = ax_risk.bar(risk_counts.index, risk_counts.values,
                           color=colors_risk, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, risk_counts.values):
            ax_risk.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(risk_counts.values) * 0.01,
                f'{int(val):,}',
                ha='center', fontsize=10, fontweight='bold'
            )
        ax_risk.set_xlabel("Risk Level")
        ax_risk.set_ylabel("Jumlah Siswa")
        ax_risk.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_risk)

    with col_ew2:
        # Boxplot RiskScore vs FinalGrade (validasi)
        st.subheader("Validasi: RiskScore vs FinalGrade")
        fig_val, ax_val = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            x='FinalGrade', y='RiskScore',
            data=filtered_df, palette='RdYlGn_r', ax=ax_val
        )
        ax_val.set_xlabel("FinalGrade (0=Terbaik, 3=Terendah)")
        ax_val.set_ylabel("RiskScore")
        ax_val.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_val)

    # Tabel siswa kritis
    st.subheader("Daftar Siswa Berisiko Kritis")
    df_kritis = filtered_df[filtered_df['RiskLevel'] == 'Kritis'].sort_values(
        'RiskScore', ascending=False
    )
    if not df_kritis.empty:
        cols_tampil = [c for c in ['RiskScore', 'RiskLevel', 'AssignmentCompletion',
                                    'Attendance', 'StressLevel', 'Motivation',
                                    'StudyHours', 'FinalGrade'] if c in df_kritis.columns]
        st.dataframe(df_kritis[cols_tampil].reset_index(drop=True), use_container_width=True)
    else:
        st.success("Tidak ada siswa dengan Risk Level Kritis pada filter yang dipilih.")

else:
    st.info("Kolom RiskScore / RiskLevel tidak ditemukan. Pastikan menggunakan data_AktivitasBelajar.csv dari notebook terbaru.")

st.divider()

# --- BARIS 6: PROFIL AKTIVITAS BELAJAR (dari Binning & Rasio EDA 1) ---
st.header("4. Profil Aktivitas Belajar — Binning & Rasio")

if 'StudyHours_Category' in filtered_df.columns:
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        # Distribusi kategori jam belajar
        st.subheader("Distribusi Kategori Jam Belajar")
        order_study = ['Rendah (<10 jam)', 'Sedang (10-20 jam)',
                       'Tinggi (20-30 jam)', 'Sangat Tinggi (>30 jam)']
        counts_study = (
            filtered_df['StudyHours_Category']
            .value_counts()
            .reindex(order_study)
            .fillna(0)
        )
        fig_sh, ax_sh = plt.subplots(figsize=(6, 4))
        ax_sh.bar(counts_study.index, counts_study.values,
                  color='#4e79a7', edgecolor='white')
        ax_sh.set_xlabel("Kategori Jam Belajar")
        ax_sh.set_ylabel("Jumlah Siswa")
        ax_sh.tick_params(axis='x', rotation=15)
        ax_sh.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_sh)

    with col_p2:
        # Engagement Score per Attendance Category
        st.subheader("Engagement Score per Kategori Kehadiran")
        if 'EngagementScore' in filtered_df.columns and 'Attendance_Category' in filtered_df.columns:
            order_att = ['Kritis (<60%)', 'Rendah (60-75%)',
                         'Baik (75-90%)', 'Sempurna (>90%)']
            fig_eng, ax_eng = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x='Attendance_Category', y='EngagementScore',
                data=filtered_df, order=order_att,
                palette='Greens', ax=ax_eng
            )
            ax_eng.set_xlabel("Kategori Kehadiran")
            ax_eng.set_ylabel("Engagement Score")
            ax_eng.tick_params(axis='x', rotation=15)
            ax_eng.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig_eng)

    col_p3, col_p4 = st.columns(2)

    with col_p3:
        # Study Efficiency per Assignment Category
        st.subheader("Efisiensi Belajar per Kategori Tugas")
        if 'StudyEfficiency' in filtered_df.columns and 'AssignmentCompletion_Category' in filtered_df.columns:
            order_assign = ['Risiko Tinggi (<60%)', 'Perlu Perhatian (60-75%)',
                            'Cukup Baik (75-90%)', 'Excellent (>90%)']
            fig_eff, ax_eff = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x='AssignmentCompletion_Category', y='StudyEfficiency',
                data=filtered_df, order=order_assign,
                palette='Blues', ax=ax_eff
            )
            ax_eff.set_xlabel("Kategori Penyelesaian Tugas")
            ax_eff.set_ylabel("Study Efficiency")
            ax_eff.tick_params(axis='x', rotation=15)
            ax_eff.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig_eff)

    with col_p4:
        # StudyStressBalance per StressLevel Category
        st.subheader("Keseimbangan Belajar-Stres")
        if 'StudyStressBalance' in filtered_df.columns and 'StressLevel_Category' in filtered_df.columns:
            order_stress = ['Rendah (1-3)', 'Sedang (4-6)', 'Tinggi (7-10)']
            fig_bal, ax_bal = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x='StressLevel_Category', y='StudyStressBalance',
                data=filtered_df, order=order_stress,
                palette='Oranges_r', ax=ax_bal
            )
            ax_bal.set_xlabel("Kategori Stres")
            ax_bal.set_ylabel("Study Stress Balance")
            ax_bal.tick_params(axis='x', rotation=15)
            ax_bal.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig_bal)

else:
    st.info("Kolom binning tidak ditemukan. Pastikan menggunakan data_AktivitasBelajar.csv dari notebook terbaru.")

st.divider()

# --- BARIS 7: KESIAPAN DIGITAL vs RISK LEVEL ---
st.header("5. Kesiapan Digital vs Risk Level")

if 'DigitalReadiness_Label' in filtered_df.columns and 'RiskLevel' in filtered_df.columns:
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        # Grouped bar: DigitalReadiness vs RiskLevel
        st.subheader("Distribusi Risk Level per Kesiapan Digital")
        digital_order = ['Tidak Siap', 'Sebagian', 'Siap Digital']
        risk_order    = ['Aman', 'Perhatian', 'Kritis']
        cross = (
            filtered_df
            .groupby(['DigitalReadiness_Label', 'RiskLevel'], observed=True)
            .size()
            .reset_index(name='count')
        )
        pivot = (
            cross
            .pivot(index='DigitalReadiness_Label', columns='RiskLevel', values='count')
            .reindex(index=digital_order, columns=risk_order)
            .fillna(0)
        )
        fig_dig, ax_dig = plt.subplots(figsize=(6, 4))
        pivot.plot(
            kind='bar', ax=ax_dig,
            color=['#2ecc71', '#f39c12', '#e74c3c'],
            edgecolor='white'
        )
        ax_dig.set_xlabel("Digital Readiness")
        ax_dig.set_ylabel("Jumlah Siswa")
        ax_dig.tick_params(axis='x', rotation=0)
        ax_dig.legend(title='Risk Level')
        ax_dig.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_dig)

    with col_d2:
        # Distribusi DigitalReadiness_Label
        st.subheader("Proporsi Kesiapan Digital Siswa")
        dig_counts = (
            filtered_df['DigitalReadiness_Label']
            .value_counts()
            .reindex(digital_order)
            .fillna(0)
        )
        fig_dp, ax_dp = plt.subplots(figsize=(6, 4))
        ax_dp.pie(
            dig_counts.values,
            labels=dig_counts.index,
            autopct='%1.1f%%',
            colors=['#e74c3c', '#f39c12', '#2ecc71'],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        ax_dp.set_title("Proporsi Kesiapan Digital")
        st.pyplot(fig_dp)

else:
    st.info("Kolom DigitalReadiness_Label / RiskLevel tidak ditemukan. Pastikan menggunakan data_AktivitasBelajar.csv dari notebook terbaru.")

st.divider()

# --- BARIS 8: DATA EXPLORER ---
st.header("6. Data Explorer — Semua Fitur")
st.markdown("Tabel interaktif lengkap.")

cols_explorer = [c for c in [
    'FinalGrade', 'RiskLevel', 'RiskScore',
    'AssignmentCompletion', 'AssignmentCompletion_Category',
    'Attendance', 'Attendance_Category',
    'StudyHours', 'StudyHours_Category',
    'StressLevel', 'StressLevel_Category',
    'EngagementScore', 'StudyEfficiency',
    'StudyStressBalance', 'DigitalReadiness_Label',
    'LearningStyle', 'EduTech', 'Motivation'
] if c in filtered_df.columns]

st.dataframe(filtered_df[cols_explorer].reset_index(drop=True), use_container_width=True)
st.caption(f"Menampilkan {len(filtered_df):,} baris dari total {len(df):,} data.")