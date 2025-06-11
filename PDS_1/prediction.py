import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier


with open('best_model_gradboost.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

with open('scaler.pkl', 'rb') as f_scaler:
    scaler = pickle.load(f_scaler)

with open('imputer.pkl', 'rb') as f_imputer:
    imputer = pickle.load(f_imputer)

with open('feature_columns.pkl', 'rb') as f_cols:
    feature_columns = pickle.load(f_cols)


def proses_data(data_input, feature_columns, imputer, scaler):
    # Menyesuaikan urutan dan melengkapi kolom yang hilang
    data_aligned = data_input.reindex(columns=feature_columns, fill_value=np.nan)

    # Imputasi nilai yang hilang
    data_imputed = pd.DataFrame(imputer.transform(data_aligned), columns=feature_columns)

    # Scaling fitur
    data_scaled = scaler.transform(data_imputed)

    return data_scaled


def prediksi_attrition(data_baru_scaled, model):
    prediksi = model.predict(data_baru_scaled)
    probabilitas = model.predict_proba(data_baru_scaled)[:, 1]
    return int(prediksi[0]), round(probabilitas[0], 4)

# === DATA BARU YANG AKAN DIPREDIKSI ===
data_baru = pd.DataFrame([{
    'Age': 38,
    'DailyRate': 1374,
    'DistanceFromHome': 18,
    'Education': 3,
    'EnvironmentSatisfaction': 2,
    'HourlyRate': 73,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobSatisfaction': 2,
    'MonthlyIncome': 8652,
    'MonthlyRate': 19234,
    'NumCompaniesWorked': 5,
    'PercentSalaryHike': 17,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 4,
    'StockOptionLevel': 2,
    'TotalWorkingYears': 12,
    'TrainingTimesLastYear': 4,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 6,
    'YearsInCurrentRole': 4,
    'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': 4,
    'BusinessTravel_Travel_Frequently': 1,
    'BusinessTravel_Travel_Rarely': 0,
    'Department_Research & Development': 1,
    'Department_Sales': 0,
    'EducationField_Life Sciences': 1,
    'EducationField_Marketing': 0,
    'EducationField_Medical': 0,
    'EducationField_Other': 0,
    'EducationField_Technical Degree': 0,
    'Gender_Male': 0,
    'JobRole_Human Resources': 0,
    'JobRole_Laboratory Technician': 0,
    'JobRole_Manager': 1,
    'JobRole_Manufacturing Director': 0,
    'JobRole_Research Director': 0,
    'JobRole_Research Scientist': 0,
    'JobRole_Sales Executive': 0,
    'JobRole_Sales Representative': 1,
    'MaritalStatus_Married': 1,
    'MaritalStatus_Single': 0,
    'OverTime_Yes': 0,
    'StabilityInRole': 0.73,
    'LoyaltyToManager': 0.81,
    'AvgTrainingPerYear': 1.46,
    'AgeWhenStarted': 23,
    'AvgYearsPerCompany': 2.4,
    'IncomePerKm': 109.5,
    'CompanyLoyalty': 0.62,
    'PromotionFrequency': 1.75,
    'AvgMonthlyIncomePerYear': 786.4
}])

# === PROSES DAN PREDIKSI ===
data_baru_scaled = proses_data(data_baru, feature_columns, imputer, scaler)
hasil_prediksi, hasil_probabilitas = prediksi_attrition(data_baru_scaled, model)

# === OUTPUT ===
print("Prediksi Attrition:", hasil_prediksi)
print("Probabilitas Attrition:", hasil_probabilitas)
