import streamlit as st
import pandas as pd
import pickle
import os
import sklearn

# Ambil direktori saat ini (yaitu tempat app.py berada)
current_dir = os.path.dirname(__file__)
models_dir = os.path.join(current_dir, 'models')

# Path ke file dalam subfolder 'models'
model_path = os.path.join(models_dir, 'best_model_RandomForest.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')
cols_path = os.path.join(models_dir, 'feature_columns.pkl')

# Load file
with open(model_path, 'rb') as f_model:
    model = pickle.load(f_model)

with open(scaler_path, 'rb') as f_scaler:
    scaler = pickle.load(f_scaler)

with open(cols_path, 'rb') as f_cols:
    feature_columns = pickle.load(f_cols)


st.set_page_config(page_title="Dropout Prediction", layout="centered")
st.title("📊 Student Dropout Risk Predictor")
st.markdown("Aplikasi memprediksi mahasiswa dropout")

# Form Input
with st.form("dropout_form"):
    st.header("Data Mahasiswa")

    col1, col2, col3 = st.columns(3)

    # ===== Kolom 1: Data Pribadi & Pendaftaran =====
    with col1:
        st.subheader("🧍 Data Pribadi")
        marital_status_map = {
            "Single": 1,
            "Married": 2,
            "Widowed": 3,
            "Divorced": 4,
            "Facto Union": 5,
            "Legally Separated": 6
        }
        marital_status_label = st.selectbox("Status Pernikahan", list(marital_status_map.keys()), index=3)
        marital_status = marital_status_map[marital_status_label]
        
        gender_map = {"Perempuan": 0, "Laki-laki": 1}
        gender_label = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
        gender = gender_map[gender_label]

        age = st.number_input("Usia Saat Mendaftar", 15, 100, 30, key="age")
        
        local_map = {"Bukan Lokal": 0, "Lokal": 1}
        is_local_label = st.selectbox("Domisili Lokal?", list(local_map.keys()))
        is_local = local_map[is_local_label]

        int_map = {"Bukan Internasional": 0, "Internasional": 1}
        international_label = st.selectbox("Mahasiswa Internasional?", list(int_map.keys()), index=0)
        international = int_map[international_label]

        displaced_map = {"Tidak": 0, "Ya": 1}
        displaced_label = st.selectbox("Status Pengungsi?", list(displaced_map.keys()), index=1)
        displaced = displaced_map[displaced_label]

        special_map = {"Tidak": 0, "Ya": 1}
        special_needs_label = st.selectbox("Berkebutuhan Khusus?", list(special_map.keys()), index=0)
        special_needs = special_map[special_needs_label]

        parents_work_map = {"Tidak Keduanya": 0, "Keduanya Bekerja": 1}
        parents_work_label = st.selectbox("Kedua Orang Tua Bekerja?", list(parents_work_map.keys()), index=2)
        parents_work = parents_work_map[parents_work_label]

        st.subheader("🎓 Pendaftaran")
        application_mode = st.number_input("Mode Pendaftaran", 0, 20, 1, key="application_mode")
        
        application_order = st.number_input("Urutan Pilihan", 0, 10, 3, key="application_order")
        
        attendance_map = {"Evening": 0, "Daytime": 1}
        attendance_label = st.selectbox("Jenis Kehadiran", list(attendance_map.keys()), index=0)
        attendance = attendance_map[attendance_label]

        previous_qualification = st.number_input("Kualifikasi Sebelumnya", 0, 20, 1, key="previous_qualification")
       
        previous_grade = st.number_input("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 90.0, key="previous_grade")
       
        admission_grade = st.number_input("Nilai Masuk", 0.0, 200.0, 80.0, key="admission_grade")

    # ===== Kolom 2: Data Akademik Semester =====
    with col2:
        st.subheader("📚 Semester 1")
        sem1_enrolled = st.number_input("Mata Kuliah Diambil (S1)", 0, 20, 6, key="sem1_enrolled")
        sem1_evals = st.number_input("Jumlah Evaluasi (S1)", 0, 20, 4, key="sem1_evals")
        sem1_approved = st.number_input("Lulus (S1)", 0, 20, 3, key="sem1_approved")
        sem1_grade = st.number_input("Nilai Rata-rata (S1)", 0.0, 20.0, 9.0, key="sem1_grade")
        sem1_credited = st.number_input("SKS Diterima (S1)", 0, 20, 0, key="sem1_credited")
        sem1_wo_eval = st.number_input("Tanpa Evaluasi (S1)", 0, 10, 0, key="sem1_wo_eval")

        st.subheader("📘 Semester 2")
        sem2_enrolled = st.number_input("Mata Kuliah Diambil (S2)", 0, 20, 4, key="sem2_enrolled")
        sem2_evals = st.number_input("Jumlah Evaluasi (S2)", 0, 20, 3, key="sem2_evals")
        sem2_approved = st.number_input("Lulus (S2)", 0, 20, 2, key="sem2_approved")
        sem2_grade = st.number_input("Nilai Rata-rata (S2)", 0.0, 20.0, 8.0, key="sem2_grade")
        sem2_credited = st.number_input("SKS Diterima (S2)", 0, 20, 0, key="sem2_credited")
        sem2_wo_eval = st.number_input("Tanpa Evaluasi (S2)", 0, 10, 0, key="sem2_wo_eval")

    # ===== Kolom 3: Sosial Ekonomi & Orang Tua =====
    with col3:
        st.subheader("💰 Sosial Ekonomi")
        debtor_map = {"Bukan Penunggak": 0, "Penunggak": 1}
        debtor_label = st.selectbox("Penunggak Biaya?", list(debtor_map.keys()))
        debtor = debtor_map[debtor_label]

        tuition_paid_map = {"Belum Terbayar": 0, "Sudah Terbayar": 1}
        tuition_paid_label = st.selectbox("Biaya Kuliah Terbayar?", list(tuition_paid_map.keys()), index=0)
        tuition_paid = tuition_paid_map[tuition_paid_label]

        scholarship_map = {"Bukan Penerima": 0, "Penerima": 1}
        scholarship_holder_label = st.selectbox("Penerima Beasiswa?", list(scholarship_map.keys()), index=0)
        scholarship_holder = scholarship_map[scholarship_holder_label]

        unemployment = st.number_input("Tingkat Pengangguran (%)", 0.0, 100.0, 12.0, key="unemployment")
        inflation = st.number_input("Tingkat Inflasi (%)", -10.0, 100.0, 5.0, key="inflation")
        gdp = st.number_input("GDP (miliar USD)", 0.0, 1000.0, 80.0, key="gdp")
        course_group = st.number_input("Kelompok Jurusan", 0, 10, 1, key="course_group")

        st.subheader("👨‍👩‍👧 Orang Tua")
        mother_edu = st.number_input("Pendidikan Ibu", 0, 10, 2, key="mother_edu")
        father_edu = st.number_input("Pendidikan Ayah", 0, 10, 3, key="father_edu")
        mother_job = st.number_input("Pekerjaan Ibu", 0, 20, 10, key="mother_job")
        father_job = st.number_input("Pekerjaan Ayah", 0, 20, 10, key="father_job")
        edu_gap = st.number_input("Selisih Pendidikan Ortu", -10, 10, -1, key="edu_gap")

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        input_dict = {
            "Marital_status": marital_status,
            "Application_mode": application_mode,
            "Application_order": application_order,
            "Daytime_evening_attendance": attendance,
            "Previous_qualification": previous_qualification,
            "Previous_qualification_grade": previous_grade,
            "Admission_grade": admission_grade,
            "Displaced": displaced,
            "Educational_special_needs": special_needs,
            "Debtor": debtor,
            "Tuition_fees_up_to_date": tuition_paid,
            "Gender": gender,
            "Scholarship_holder": scholarship_holder,
            "Age_at_enrollment": age,
            "International": international,
            "Curricular_units_1st_sem_enrolled": sem1_enrolled,
            "Curricular_units_1st_sem_evaluations": sem1_evals,
            "Curricular_units_1st_sem_approved": sem1_approved,
            "Curricular_units_1st_sem_grade": sem1_grade,
            "Curricular_units_1st_sem_credited": sem1_credited,
            "Curricular_units_1st_sem_without_evaluations": sem1_wo_eval,
            "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
            "Curricular_units_2nd_sem_evaluations": sem2_evals,
            "Curricular_units_2nd_sem_approved": sem2_approved,
            "Curricular_units_2nd_sem_grade": sem2_grade,
            "Curricular_units_2nd_sem_credited": sem2_credited,
            "Curricular_units_2nd_sem_without_evaluations": sem2_wo_eval,
            "Unemployment_rate": unemployment,
            "Inflation_rate": inflation,
            "GDP": gdp,
            "Course_group": course_group,
            "Is_local": is_local,
            "Mother_edu_level": mother_edu,
            "Father_edu_level": father_edu,
            "Mother_job": mother_job,
            "Father_job": father_job,
            "Parental_education_gap": edu_gap,
            "Is_both_parents_employed": parents_work
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        scaled_input = scaler.transform(input_df)
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        if pred == 1:
            st.error("🔴 **Prediksi: Berisiko Dropout**")
            st.markdown("🚨 Mahasiswa ini memiliki kemungkinan **tinggi** untuk mengalami *dropout*. Perlu perhatian lebih lanjut.")
        else:
            st.success("🟢 **Prediksi: Tidak Dropout**")
            st.markdown("✅ Mahasiswa ini diprediksi akan **bertahan** dalam studinya.")
        
        # Visualisasi probabilitas
        st.subheader("📈 Probabilitas Dropout")
        st.progress(int(prob * 100))
        
        # Tambahkan label persentase
        if prob >= 0.75:
            risk_level = "⚠️ Sangat Tinggi"
        elif prob >= 0.5:
            risk_level = "🟠 Tinggi"
        elif prob >= 0.25:
            risk_level = "🟡 Sedang"
        else:
            risk_level = "🟢 Rendah"
        
        st.markdown(f"**Probabilitas:** `{prob:.2%}` ({risk_level})")

        if prob > 0.6:
            st.warning("📌 **Saran:** Konseling akademik dan dukungan sosial disarankan.")
        elif prob > 0.3:
            st.info("📌 **Saran:** Pantau perkembangan akademik mahasiswa ini secara berkala.")
        else:
            st.success("📌 **Saran:** Tidak ada tindakan khusus diperlukan saat ini.")

