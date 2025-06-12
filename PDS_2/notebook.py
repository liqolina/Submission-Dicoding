- Nama:Lutfi Aundrie Hermawan
- Email:lutfiaunher@gmail.com
- Id Dicoding:lutfi_hermawan

## Persiapan

### Menyiapkan library yang dibutuhkan

1. Visualisasi dan Manipulasi Data
2. Preprocessing dan Pemodelan
3. Evaluasi Model
4. Save Model dan Database
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sqlalchemy import create_engine
import pickle

from sqlalchemy import create_engine

import joblib

"""### Menyiapkan data yang akan digunakan

Link dataset dapat diakses dibawah ini

[Github](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

## Data Understanding

Dataset yang telah disiapkan berisi informasi mengenai mahasiswa dari Jaya Jaya Institut. Terdapat 37 Kolom dan 4424 Baris. Tujuan dataset ini untuk memprediksi status mahasiswa akan dropout, masih aktif, atau lulus. Berikut masing-masing deskripsi kolom dibawah ini

| Column name | Description |
| --- | --- |
|Marital status | The marital status of the student. (Categorical) 1 – single 2 – married 3 – widower 4 – divorced 5 – facto union 6 – legally separated |
| Application mode | The method of application used by the student. (Categorical) 1 - 1st phase - general contingent 2 - Ordinance No. 612/93 5 - 1st phase - special contingent (Azores Island) 7 - Holders of other higher courses 10 - Ordinance No. 854-B/99 15 - International student (bachelor) 16 - 1st phase - special contingent (Madeira Island) 17 - 2nd phase - general contingent 18 - 3rd phase - general contingent 26 - Ordinance No. 533-A/99, item b2) (Different Plan) 27 - Ordinance No. 533-A/99, item b3 (Other Institution) 39 - Over 23 years old 42 - Transfer 43 - Change of course 44 - Technological specialization diploma holders 51 - Change of institution/course 53 - Short cycle diploma holders 57 - Change of institution/course (International)|
|Application order | The order in which the student applied. (Numerical) Application order (between 0 - first choice; and 9 last choice) |
|Course | The course taken by the student. (Categorical) 33 - Biofuel Production Technologies 171 - Animation and Multimedia Design 8014 - Social Service (evening attendance) 9003 - Agronomy 9070 - Communication Design 9085 - Veterinary Nursing 9119 - Informatics Engineering 9130 - Equinculture 9147 - Management 9238 - Social Service 9254 - Tourism 9500 - Nursing 9556 - Oral Hygiene 9670 - Advertising and Marketing Management 9773 - Journalism and Communication 9853 - Basic Education 9991 - Management (evening attendance)|
|Daytime/evening attendance | Whether the student attends classes during the day or in the evening. (Categorical) 1 – daytime 0 - evening |
|Previous qualification| The qualification obtained by the student before enrolling in higher education. (Categorical) 1 - Secondary education 2 - Higher education - bachelor's degree 3 - Higher education - degree 4 - Higher education - master's 5 - Higher education - doctorate 6 - Frequency of higher education 9 - 12th year of schooling - not completed 10 - 11th year of schooling - not completed 12 - Other - 11th year of schooling 14 - 10th year of schooling 15 - 10th year of schooling - not completed 19 - Basic education 3rd cycle (9th/10th/11th year) or equiv. 38 - Basic education 2nd cycle (6th/7th/8th year) or equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 42 - Professional higher technical course 43 - Higher education - master (2nd cycle) |
|Previous qualification (grade) | Grade of previous qualification (between 0 and 200) |
| Nacionality | The nationality of the student. (Categorical) 1 - Portuguese; 2 - German; 6 - Spanish; 11 - Italian; 13 - Dutch; 14 - English; 17 - Lithuanian; 21 - Angolan; 22 - Cape Verdean; 24 - Guinean; 25 - Mozambican; 26 - Santomean; 32 - Turkish; 41 - Brazilian; 62 - Romanian; 100 - Moldova (Republic of); 101 - Mexican; 103 - Ukrainian; 105 - Russian; 108 - Cuban; 109 - Colombian|
|Mother's qualification | The qualification of the student's mother. (Categorical) 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)|
|Father's qualification | The qualification of the student's father. (Categorical) 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 13 - 2nd year complementary high school course 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 20 - Complementary High School Course 22 - Technical-professional course 25 - Complementary High School Course - not concluded 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 31 - General Course of Administration and Commerce 33 - Supplementary Accounting and Administration 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle) |
| Mother's occupation | The occupation of the student's mother. (Categorical) 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 122 - Health professionals 123 - teachers 125 - Specialists in information and communication technologies (ICT) 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 171 - Skilled construction workers and the like, except electricians 173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like 175 - Workers in food processing, woodworking, clothing and other industries and crafts 191 - cleaning workers 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants |
| Father's occupation | The occupation of the student's father. (Categorical) 0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 101 - Armed Forces Officers 102 - Armed Forces Sergeants 103 - Other Armed Forces personnel 112 - Directors of administrative and commercial services 114 - Hotel, catering, trade and other services directors 121 - Specialists in the physical sciences, mathematics, engineering and related techniques 122 - Health professionals 123 - teachers 124 - Specialists in finance, accounting, administrative organization, public and commercial relations 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 135 - Information and communication technology technicians 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 154 - Protection and security services personnel 161 - Market-oriented farmers and skilled agricultural and animal production workers 163 - Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence 171 - Skilled construction workers and the like, except electricians 172 - Skilled workers in metallurgy, metalworking and similar 174 - Skilled workers in electricity and electronics 175 - Workers in food processing, woodworking, clothing and other industries and crafts 181 - Fixed plant and machine operators 182 - assembly workers 183 - Vehicle drivers and mobile equipment operators 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants 195 - Street vendors (except food) and street service providers |
| Admission grade | Admission grade (between 0 and 200) |
| Displaced | Whether the student is a displaced person. (Categorical) 	1 – yes 0 – no |
| Educational special needs | Whether the student has any special educational needs. (Categorical) 1 – yes 0 – no |
|Debtor | Whether the student is a debtor. (Categorical) 1 – yes 0 – no|
|Tuition fees up to date | Whether the student's tuition fees are up to date. (Categorical) 1 – yes 0 – no|
|Gender | The gender of the student. (Categorical) 1 – male 0 – female |
|Scholarship holder | Whether the student is a scholarship holder. (Categorical) 1 – yes 0 – no |
|Age at enrollment | The age of the student at the time of enrollment. (Numerical)|
|International | Whether the student is an international student. (Categorical) 1 – yes 0 – no|
|Curricular units 1st sem (credited) | The number of curricular units credited by the student in the first semester. (Numerical) |
| Curricular units 1st sem (enrolled) | The number of curricular units enrolled by the student in the first semester. (Numerical) |
| Curricular units 1st sem (evaluations) | The number of curricular units evaluated by the student in the first semester. (Numerical) |
| Curricular units 1st sem (approved) | The number of curricular units approved by the student in the first semester. (Numerical) |

Memuat dataset ke dalam DataFrame menggunakan pustaka Pandas, kemudian menampilkan ringkasan dari DataFrame tersebut
"""

df = pd.read_csv('/content/data.csv', sep=';')
df.info()

df.describe(include='all')

"""## Data Preparation / Preprocessing

###Data Preparation

Berikut menampilkan 10 baris pertama dari dataset df. Fungsi ini digunakan untuk memberikan informasi yang ada di dataset.
"""

df.head(10)

"""#### Check Missing Values

Digunakan untuk mengecek apabila terdapat nilai yang hilang pada dataset.
"""

missing_num = df[df.columns].isna().sum().sort_values(ascending=False)
missing_perc = (df[df.columns].isna().sum()/len(df)*100).sort_values(ascending=False)
missing = pd.concat([missing_num,missing_perc],keys=['Total','Percentage'],axis=1)
missing_train = missing[missing['Percentage']>=0]
missing_train

"""Berdasarkan hasil pengecekan nilai yang hilang (missing values), tidak ditemukan data missing values, sehingga tidak perlu melanjutkan langkah penghapusan baris/kolom atau Imputant.

#### Check Duplicate Values

Digunakan untuk memeriksa data apabila terdapat 2 atau lebih data yang sama.
"""

num_duplicates = df.duplicated().sum()
percent_duplicates = (num_duplicates / len(df)) * 100

print(f"Jumlah data duplikat: {num_duplicates}")
print(f"Persentase data duplikat: {percent_duplicates:.2f}%")

"""Berdasarkan hasil pengecekan nilai yang terduplikat (Duplicate Values), tidak ditemukan data yang duplikat.

####Nilai Unik pada Kolom Kategorikal

Bertujuan untuk memeriksa nilai-nilai unik yang ada di dalam daftar kolom yang diidentifikasi sebagai kolom kategorikal.
"""

categorical_columns = [
    'Marital_status',
    'Application_mode',
    'Application_order',
    'Course',
    'Daytime_evening_attendance',
    'Previous_qualification',
    'Nacionality',
    'Mothers_qualification',
    'Fathers_qualification',
    'Mothers_occupation',
    'Fathers_occupation',
    'Displaced',
    'Educational_special_needs',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Gender',
    'Scholarship_holder',
    'International',
    'Status'
]

for column in categorical_columns:
    unique_values = df[column].unique()
    num_unique = df[column].nunique()

    print(f"Nama Kolom: {column}")
    print(f"Jumlah nilai unik: {num_unique}")
    print(f"Nilai unik: {unique_values}")
    print("-" * 60)

df.info()

"""##Exploratory Data Analysis (EDA)

###Visualisasi Distribusi Data Status

Menvisualisasi distribusi data status mahasiswa saat di kampus
"""

plt.figure(figsize=(10, 6))
status_counts = df['Status'].value_counts()
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Status Mahasiswa')
plt.axis('equal')
plt.show()

"""Bedasarkan visualisasi tersebut bahwa presentasi Graduate sebesar 49.9% diikuti dengan Dropout sebesar 32.1% dan Enrolled sebesar 17.9%"""

plt.figure(figsize=(6, 4))
df['Status'].value_counts().plot(kind='bar')
plt.title('Distribusi Status Mahasiswa')
plt.xlabel('Status')
plt.ylabel('Jumlah Mahasiswa')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

"""Bedasarkan distribusi status mahasiswa bahwa jumlah yang graduate lebih dari 2000 orang, dropout kurang dari 1500 orang dan enrolled kurang dari 1000.

###Visualisasi Berdasarkan Gender

Menvisualisasi bedasarkan gender untuk mengetahui ratio antara laki-laki dan perempuan
"""

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
gender_status = df.groupby(['Gender', 'Status']).size().unstack()
gender_status.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Status Mahasiswa Berdasarkan Jenis Kelamin')
plt.xlabel('Jenis Kelamin (0: Perempuan, 1: Laki-laki)')
plt.ylabel('Jumlah Mahasiswa')
plt.xticks(rotation=0)

"""Bedasarkan jenis kelamin bahwa jenis kelamin laki laki lebih banyak mahasiswa yang lulus dibandingkan jumlah mahasiswa yang dropout

###Correlation Distribution

Memvisualisasikan korelasi antar variabel numerik dalam dataset. Korelasi mengukur seberapa kuat hubungan antara dua variabel dengan menggunakan visualisasi heatmap.
"""

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            fmt='.2f',
            cmap='coolwarm')
plt.title('Korelasi Antar Variabel Numerik')
plt.show()

"""Fitur yang paling berkorelasi dengan Status salah satunya Adminission_grade dan Previous_qualification_grade. Fitur dengan korelasi tinggi dapat mempengaruhi interpretasi model.

###Visualisasi Numerical

Membuat visualisasi histogram untuk setiap kolom numerik dalam dataset df. Tujuan ini untuk mengidentifikasi potensi outlier atau nilai ekstrem.
"""

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Histogram semua numerik
df[num_cols].hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()

"""Membuat visualisasi boxplot untuk setiap kolom numerik dalam dataset df. Tujuan ini untuk memahami distribusi data dan mengidentifikasi nilai outlier"""

cols_per_row = 3

for i, col in enumerate(num_cols):
    if i % cols_per_row == 0:
        # Mulai baris baru
        plt.figure(figsize=(15, 4))

    plt.subplot(1, cols_per_row, (i % cols_per_row) + 1)
    df.boxplot(column=col)
    plt.title(f'Boxplot Kolom {col}')

    # Jika sudah 3 plot atau terakhir, tampilkan
    if (i % cols_per_row == cols_per_row - 1) or (i == len(num_cols) - 1):
        plt.tight_layout()
        plt.show()

"""###Data Preprocessing"""

df_pre=df.copy()

"""#### Mengubah tipe data object (kolom kategorikal) ke dalam bentuk tipe category"""

# Melakukan konversi tipe data ke kategori
for column in categorical_columns:
    if column in df_pre.columns:
        df_pre[column] = df_pre[column].astype('category')
        print(f"Kolom '{column}' berhasil dikonversi ke tipe: {df[column].dtype}")
    else:
        print(f"Kolom '{column}' tidak ditemukan dalam DataFrame.")

"""####Feature Engineering

Tahapan ini digunakan untuk membuat fitur baru (feature engineering) dari data yang sudah ada dalam DataFrame df. Tujuannya adalah untuk menciptakan variabel yang lebih informatif untuk analisis dan meningkatkan performa model prediktif

- Membuat fitur avg_sem_grade sebagai rata rata nilai mata kuliah semester 1 dan semester

- total_units_approved Bertujuan untuk menganalisis performa akademik dengan jumlah total mata kuliah yang berhasil diselesaikan.
"""

# Curricular_units
df_pre['avg_sem_grade'] = (df_pre['Curricular_units_1st_sem_grade'] + df_pre['Curricular_units_2nd_sem_grade'])/2
df_pre['total_units_approved'] = (df_pre['Curricular_units_1st_sem_approved'] + df_pre['Curricular_units_2nd_sem_approved'])

# tambahkan ke list numerik
if 'num_cols' not in locals() or not isinstance(num_cols, list):
    num_cols = df_pre.select_dtypes(include=[np.number]).columns.tolist()

num_cols += ['avg_sem_grade','total_units_approved']

numeric_cols = df_pre.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'Status_num' in numeric_cols:
    numeric_cols.remove('Status_num')

"""### Labeling

#### LabelEncoder

Digunakan untuk mengubah nilai kategorikal pada kolom Status
"""

label_encoder = LabelEncoder()

df_pre['Status'] = label_encoder.fit_transform(df_pre['Status'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nMapping Label Encoder:")
print(label_mapping)

"""Graduate → 2, Dropout → 0, Enrolled → 1."""

df_clean=df_pre.copy()

"""### Splitting data into Trainand Test

Proporsi data tes adalah 30% (test_size=0.3), sedangkan data training adalah 80%.
"""

X = df_clean.drop(['Status'], axis=1)
y = df_clean['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

"""###StandardScaler

Melakukan standarisasi pada fitur-fitur numerik dalam dataset pelatihan dan pengujian
"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""## Modeling

###Logistic Regression

Melatih model Logistic Regression tanpa menggunakan GridSearchCV
"""

lr_model = LogisticRegression(C=0.01, max_iter=5000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Prediksi
y_pred_scaled = lr_model.predict(X_test_scaled)
y_prob_scaled = lr_model.predict_proba(X_test_scaled)[:, 1]

# Menghitung akurasi dari prediksi
akurasi = accuracy_score(y_test, y_pred_scaled)

print(f"Akurasi Model: {akurasi:.4f}")

"""###Random Forest"""

rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Menghitung akurasi dari prediksi
akurasi = accuracy_score(y_test, y_pred_rf)

print(f"Akurasi Model: {akurasi:.4f}")

"""###Gradient Boosting"""

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

# Resampling kelas minoritas
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Latih model pada data hasil SMOTE
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_res, y_train_res)

y_pred_gb = gb.predict(X_test_scaled)
y_prob_gb = gb.predict_proba(X_test_scaled)[:, 1]

# Menghitung akurasi dari prediksi
akurasi = accuracy_score(y_test, y_pred_gb)

print(f"Akurasi Model: {akurasi:.4f}")

"""###XGBoost"""

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.0)
xgb.fit(X_train_scaled, y_train)

y_pred_xgb = xgb.predict(X_test_scaled)
y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

# Menghitung akurasi dari prediksi
akurasi = accuracy_score(y_test, y_pred_xgb)

print(f"Akurasi Model: {akurasi:.4f}")

"""###SVM"""

svc = SVC(probability=True, class_weight='balanced', random_state=42)
svc.fit(X_train_scaled, y_train)

y_pred_svc = svc.predict(X_test_scaled)
y_prob_svc = svc.predict_proba(X_test_scaled)[:, 1]

# Menghitung akurasi dari prediksi
akurasi = accuracy_score(y_test, y_pred_svc)

print(f"Akurasi Model: {akurasi:.4f}")

"""## Evaluation

###Logistic Regression
"""

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_scaled))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_scaled))

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_scaled)

# Visualisasikan sebagai heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Tetap', 'Keluar'],   # Prediksi
            yticklabels=['Tetap', 'Keluar'])   # Aktual
plt.xlabel('Prediksi')
plt.ylabel('Kondisi Aktual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

"""|                    | **Prediksi: Tetap**  | **Prediksi: Keluar**  |
| ------------------ | -------------------- | --------------------- |
| **Aktual: Tetap**  | **TN**  | **FP**  |
| **Aktual: Keluar** | **FN**  | **TP**  |

Keterangan:
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive (TP)

Berdasarkan modeling tersebut dan hasil Confusion Matrix, dapat disimpulkan bahwa:
- Model Logistic Regression cukup efektif secara keseluruhan (akurasi 73%), namun memiliki kelemahan signifikan dalam mengklasifikasikan kelas 1, yang kemungkinan merupakan kelas minoritas atau memiliki fitur yang mirip dengan kelas lain.

###Random Forest
"""

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_rf))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Visualisasikan sebagai heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Tetap', 'Keluar'],   # Prediksi
            yticklabels=['Tetap', 'Keluar'])   # Aktual
plt.xlabel('Prediksi')
plt.ylabel('Kondisi Aktual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

"""|                    | **Prediksi: Tetap**  | **Prediksi: Keluar**  |
| ------------------ | -------------------- | --------------------- |
| **Aktual: Tetap**  | **TN**  | **FP**  |
| **Aktual: Keluar** | **FN**  | **TP**  |

Keterangan:
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive (TP)

Berdasarkan modeling tersebut dan hasil Confusion Matrix, dapat disimpulkan bahwa:
- Random Forest lebih baik daripada Logistic Regression secara keseluruhan:
-- Akurasi naik dari 73% → 78%.
-- Kelas 2 dikelola lebih baik (recall naik dari 0.79 → 0.94).

###Gradient Boosting
"""

print("=== Gradient Boosting ===")
print(classification_report(y_test, y_pred_gb))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_gb))

# Confusion matrix
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Visualisasikan sebagai heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_gb,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Tetap', 'Keluar'],   # Prediksi
            yticklabels=['Tetap', 'Keluar'])   # Aktual
plt.xlabel('Prediksi')
plt.ylabel('Kondisi Aktual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

"""|                    | **Prediksi: Tetap**  | **Prediksi: Keluar**  |
| ------------------ | -------------------- | --------------------- |
| **Aktual: Tetap**  | **TN**  | **FP**  |
| **Aktual: Keluar** | **FN**  | **TP**  |

Keterangan:
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive (TP)

Berdasarkan modeling tersebut dan hasil Confusion Matrix, dapat disimpulkan bahwa:
- Gradient Boosting memberikan performa paling seimbang:
-- Tidak hanya akurat (77%), tapi juga adil terhadap semua kelas.
-- F1-macro tertinggi (0.72) menunjukkan bahwa model tidak terlalu bias terhadap kelas dominan.

###SVM

print("=== Support Vector Machine ===")
print(classification_report(y_test, y_pred_svc))
"""

print("=== Support Vector Machine ===")
print(classification_report(y_test, y_pred_svc))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_svc))

# Confusion matrix
cm_svc = confusion_matrix(y_test, y_pred_svc)

# Visualisasikan sebagai heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_svc,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Tetap', 'Keluar'],   # Prediksi
            yticklabels=['Tetap', 'Keluar'])   # Aktual
plt.xlabel('Prediksi')
plt.ylabel('Kondisi Aktual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

"""|                    | **Prediksi: Tetap**  | **Prediksi: Keluar**  |
| ------------------ | -------------------- | --------------------- |
| **Aktual: Tetap**  | **TN**  | **FP**  |
| **Aktual: Keluar** | **FN**  | **TP**  |

Keterangan:
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- True Positive (TP)


"""



"""Berdasarkan modeling tersebut dan hasil Confusion Matrix, dapat disimpulkan bahwa:
- SVM memberikan hasil yang kurang optimal dibanding model lainnya, terutama:
-- Akurasi keseluruhan paling rendah (72%)
-- F1-macro juga paling rendah (0.68) → artinya distribusi performa antar kelas tidak seimbang.

### Perbandingan Akhir Semua Model
"""

def get_recall_for_class(report, class_label):
    """Ambil recall dari classification_report untuk label tertentu."""
    return float(report[class_label]['recall'])

# Buat dictionary untuk ringkasan
summary_data = []

# Evaluasi tiap model
models = {
    "LogisticRegression": y_pred_scaled,
    "RandomForest": y_pred_rf,
    "GradientBoosting": y_pred_gb,
    "SVM": y_pred_svc
}

for model_name, y_pred in models.items():
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall_dropout = get_recall_for_class(report, '0')  # Recall untuk kelas 'Dropout' (0)

    summary_data.append({
        'model': model_name,
        'accuracy': acc,
        'recall_dropout': recall_dropout
    })

# Buat DataFrame dari hasil
summary_df = pd.DataFrame(summary_data).set_index('model')
print("\nSummary:")
print(summary_df)

"""| Model                   | Akurasi | Recall Dropout (Kelas 0) | Interpretasi                                                                    |
| ----------------------- | ------- | ------------------------ | ------------------------------------------------------------------------------- |
| **Random Forest**       | 0.775   | 0.764                    | Akurasi tertinggi, recall dropout cukup tinggi dan stabil                       |
| **Logistic Regression** | 0.773   | **0.771**                | Recall tertinggi untuk dropout, cocok jika ingin **menghindari false negative** |
| **Gradient Boosting**   | 0.767   | 0.729                    | Paling seimbang antar kelas, namun sedikit lebih rendah untuk dropout           |
| **SVM**                 | 0.723   | 0.654                    | Performa terendah, meski recall dropout masih cukup                             |

### Pilihan Model Terbaik bedasarkan Dropout

Jika fokus utama adalah memprediksi siapa yang akan Dropout (kelas 0), maka:

- Semua model memberikan F1-score Dropout di kisaran 0.74–0.78
- Random Forest dan Gradient Boosting sedikit lebih baik (F1: 0.78)

### Pilihan Model Terbaik bedasarkan Semua Status

Jika fokus  ingin prediksi yang seimbang antar semua kategori (Dropout, Enrolled, Graduate), maka:

- Gradient Boosting adalah model terbaik
-- Macro F1 tertinggi (0.72)
-- Performa terbaik kedua untuk Dropout
-- Performa paling stabil untuk Enrolled dan Graduate

## Ekspor Model Terbaik
"""

best_name = summary_df['accuracy'].idxmax()
best_model = models[best_name]
print(f"\n>> Best model: {best_name} <<")

joblib.dump(best_model, '/content/best_model.joblib')
print("Model terbaik telah disimpan ke file: /content/best_model.joblib")

"""## Save Model

###For Requirements
"""

!pip freeze > requirements.txt

"""### For Supabase"""

df.to_csv('data_supabase.csv', index=False)

"""###Supabase and PostgreSQL"""

mahasiswa_df_spb = pd.read_csv('data_supabase.csv')

URL = "postgresql://postgres.kvbhwcsyylrwxsnryrvk:LaskarAI0193@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
engine = create_engine(URL)
mahasiswa_df_spb.to_sql('data_mahasiswa', engine)

URL = "postgresql://postgres.kvbhwcsyylrwxsnryrvk:LaskarAI0193@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
engine = create_engine(URL)
orders_df = pd.read_sql_table(table_name="data_mahasiswa", con=engine.connect())
orders_df.head(5)

df.head(10)
