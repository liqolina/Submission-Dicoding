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
# === Mapping dropdown ===
mappings = {
    'Marital_status': {1:'Single',2:'Married',3:'Widower',4:'Divorced',5:'Union',6:'Separated'},
    'Application_mode': {
        1:'1st-general', 2:'Ordinance612',5:'Special-Azores',7:'Other-courses',
        10:'Ordinance854',15:'International',16:'Special-Madeira',17:'2nd-general',
        18:'3rd-general',26:'Ordinance533-b2',27:'Ordinance533-b3',39:'Over23',
        42:'Transfer',43:'Change-course',44:'Tech-diploma',51:'Change-institution',
        53:'Short-cycle',57:'Change-international'
    },
    'Daytime_evening_attendance': {1:'Daytime',0:'Evening'},
    'Previous_qualification': {
        1:'Secondary',2:"Bachelor",3:'Degree',4:"Master",5:'Doctorate',
        6:'Frequency',9:'12th-incomplete',10:'11th-incomplete',12:'Other-11th',
        14:'10th',15:'10th-incomplete',19:'Basic-3rd',38:'Basic-2nd',
        39:'Tech-spec',40:'Higher-1st',42:'Professional',43:'Master-2nd'
    },
    'Displaced': {0:'No',1:'Yes'},
    'Educational_special_needs': {0:'No',1:'Yes'},
    'Debtor': {0:'No',1:'Yes'},
    'Tuition_fees_up_to_date': {0:'No',1:'Yes'},
    'Gender': {0:'Female',1:'Male'},
    'Scholarship_holder': {0:'No',1:'Yes'},
    'International': {0:'No',1:'Yes'},
    'Is_local': {0:'No',1:'Yes'},
    'Is_both_parents_employed': {0:'No',1:'Yes'}
}

# === Fungsi untuk merender input ===
def render_input():
    st.sidebar.header("📝 Isi Data Mahasiswa")

    inputs = {}
    for col in feature_columns:
        if col in mappings:
            keys = list(mappings[col].keys())
            values = [mappings[col][k] for k in keys]
            selected = st.sidebar.selectbox(col, keys, format_func=lambda x: mappings[col][x], key=col)
            inputs[col] = selected
        elif col in ['Previous_qualification_grade','Admission_grade',
                     'Curricular_units_1st_sem_grade','Curricular_units_2nd_sem_grade',
                     'Unemployment_rate','Inflation_rate','GDP']:
            inputs[col] = st.sidebar.number_input(col, value=0.0, key=col)
        else:
            inputs[col] = st.sidebar.number_input(col, value=0, key=col)
    return pd.DataFrame([inputs])

# === Konfigurasi App ===
st.set_page_config(page_title="Dropout Prediction", layout="wide")
st.title("🎓 Prediksi Mahasiswa Dropout")
st.markdown("Gunakan panel samping (sidebar) untuk mengisi data mahasiswa.")

# === Input dan Prediksi ===
df_input = render_input()

if st.sidebar.button("🔍 Prediksi"):
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    scaled_input = scaler.transform(df_input)
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    status = "🔴 Dropout" if pred == 1 else "🟢 Tidak Dropout"
    st.success(f"Hasil Prediksi: **{status}**")
    st.metric("Probabilitas Dropout", f"{prob:.2%}")
