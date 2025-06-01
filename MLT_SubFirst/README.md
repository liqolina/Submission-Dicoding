# Laporan Proyek Machine Learning - Lutfi Aundrie Hermawan

## Domain Proyek

Dengan pertumbuhan penduduk manusia beberapa tahun terakhir, kebutuhan dalam memenuhi tempat tinggal mengalami perkembangan, baik dari segi hunian sederhana maupun hunian mewah. Hal ini membuat hunian tidak sekadar berfungsi sebagai tempat tinggal, tetapi juga mencerminkan aspek estetika, status sosial, serta gaya hidup pemiliknya. Hunian menjadi representasi identitas individu maupun keluarga, di mana desain arsitektur, interior, hingga lokasi dipilih untuk menunjang citra dan kenyamanan.
    
Namun dengan perkembangan industri hunian yang semakin meningkat, terdapat sejumlah tantangan dalam menentukan harga hunian yang akurat dan kompetitif. Penentuan harga ini sangat mempengaruhi dampak yang ditimbulkan terhadap pasar properti secara keseluruhan. Hal ini disebabkan oleh peran harga sebagai indikator utama dalam menentukan nilai sebuah properti, yang dipengaruhi oleh berbagai aspek, mulai dari luas properti hunian, jumlah kamar, dan hal lainnya yang dapat mempengaruhi harga properti.
    
Metode tradisional seperti penilainan manual mengadapi keterbatasan dalam memahami hubungan yang lebih kompleks antara berbagai variable yang dapat mempengaruhi harga. Namun dengan teknologi semakin maju seperti machine learning, dan kemampuannya dalam mengolah data dalam jumlah besar serta mengenali pola-pola tersembunyi di dalam data. Machine learning mampu memberikan prediksi harga yang lebih akurat dan adaptif terhadap dinamika pasar. 
    
Proyek Machine Learning ini, digunakan sejumlah algoritma machine learning seperti Ridge Regression, Lasso Regression, ElasticNet, dan Support Vector Regression (SVR) untuk membangun model prediktif harga properti. Pemilihan algoritma-algoritma tersebut didasarkan pada kemampuannya dalam menangani permasalahan  multikolinearitas antar variabel independen serta penerapan teknik regularisasi yang efektif. Pendekatan ini tidak hanya membantu mengurangi risiko overfitting, tetapi juga meningkatkan kemampuan model dalam melakukan prediksi terhadap data baru secara lebih andal dan konsisten.
    
Dalam studi ini, Xin dan Khalid membandingkan kinerja model Ridge Regression dan Lasso Regression dalam memprediksi harga rumah di Ames, Iowa, menggunakan data dari tahun 2006 hingga 2010. Kedua model ini dipilih karena kemampuannya dalam mengatasi multikolinearitas, yang sering terjadi dalam analisis multivariat. Evaluasi model dilakukan menggunakan Root Mean Square Error (RMSE) dan adjusted R-squared. Hasilnya menunjukkan bahwa model Lasso Regression memberikan performa yang lebih baik dibandingkan dengan Ridge Regression. Variabel yang dipilih dalam model ini mencakup ukuran rumah, usia rumah, kondisi rumah, dan lokasi rumah. 

Referensi : Xin, S. J., & Khalid, K. (2018). Modelling House Price Using Ridge Regression and Lasso Regression. International Journal of Engineering & Technology, 7(4.30), 498-501. [https://doi.org/10.14419/ijet.v7i4.30.22378](https://doi.org/10.14419/ijet.v7i4.30.22378)

## Business Understanding

### Problem Statements
Nilai suatu hunian dipengaruhi oleh berbagai faktor seperti luas bangunan, jumlah kamar, dan elemen lainnya yang turut menentukan harga. Kompleksitas faktor-faktor tersebut menjadikan penetapan harga properti sebagai sebuah tantangan sulit di pasar properti hunian. Oleh karena itu,terdapat beberapa pernyataan masalah utama

1. Bagaimana memprediksi harga rumah yang mampu mengakomodasi berbagai variabel kompleks secara akurat?
2. Bagaimana mengatasi permasalahan multikolinearitas dan overfitting dalam prediksi harga rumah?
3. Bagaimana pemilihan fitur yang relevan dalam membantu memprediksi harga hunian ?

### Goals
Dari pernyataan masalah tersebut, tujuan proyek ini dapat dirumuskan sebagai berikut:

1. Menggunakan model machine learning yang dapat mengolah berbagai fitur kompleks seperti luas bangunan, jumlah kamar, lokasi, dan usia properti. Algoritma seperti Ridge Regression, Lasso Regression, ElasticNet, dan Support Vector Regression (SVR) digunakan karena dapat mengenali pola-pola non-linear dan menghasilkan prediksi yang lebih akurat dibanding metode tradisional.
2. Mengurangi overfitting dan multikolinearitas diatasi dengan penggunaan algoritma regularisasi seperti Ridge dan Lasso Regression dan cross-validation juga digunakan untuk mengevaluasi performa model tetap baik di data baru.
3. Pemilihan fitur dilakukan melalui proses feature selection dengan bantuan algoritma machine learning seperti Lasso Regression, yang secara otomatis mengeliminasi fitur yang tidak signifikan. Pemilihan fitur ini membantu meningkatkan akurasi model dan mengurangi risiko overfitting.

### Solution statements
Guna mencapai tujuan tersebut, diterapkan strategi solusi dengan mengandalkan variasi model machine learning.

1. Menggunakan beberapa algoritma seperti Ridge, Lasso, ElasticNet, dan SVR. Ridge Regression berfungsi untuk mengatasi overfitting dengan menambahkan penalti guna menekan kompleksitas model. Lasso Regression cenderung menyederhanakan model dengan mengeliminasi fitur-fitur yang kurang relevan. ElasticNet merupakan metode gabungan yang memanfaatkan keunggulan dari Ridge dan Lasso secara bersamaan. Sementara itu, SVR (Support Vector Regression) diterapkan pada data yang mengandung banyak gangguan (noise) atau memiliki pola hubungan yang tidak bersifat linear.
2. Meningkatkan akurasi dan performa model dengan Hyperparameter Tuning dengan menggunakan metode GridSearchCV. Teknik ini secara sistematis mengeksplorasi berbagai kombinasi parameter untuk menemukan konfigurasi terbaik, dengan mengacu pada metrik evaluasi seperti Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE) sebagai acuan performa.
3. Menggunakan metrik evaluasi seperti RMSE dan MSE. Semakin rendah nilai RMSE, semakin baik model dalam memprediksi nilai yang mendekati nilai aktual. 

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah USA Housing Dataset yang diperoleh dari platform Kaggle. Dataset ini berisi informasi mengenai sejumlah fitur yang mempengaruhi harga rumah di Amerika Serikat. Dataset tersebut tersedia secara publik dan dapat diunduh melalui Kaggle [USA Housing Dataset](https://www.kaggle.com/datasets/fratzcan/usa-house-prices). Berikut ini adalah deskripsi rinci dari setiap fitur pada dataset: 

### Variabel-variabel pada USA Housing Dataset adalah sebagai berikut:
- Date: The date when the property was sold. This feature helps in understanding the temporal trends in property prices.
- Price:The sale price of the property in USD. This is the target variable we aim to predict.
- Bedrooms:The number of bedrooms in the property. Generally, properties with more bedrooms tend to have higher prices.
- Bathrooms: The number of bathrooms in the property. Similar to bedrooms, more bathrooms can increase a property’s value.
- Sqft Living: The size of the living area in square feet. Larger living areas are typically associated with higher property values.
- Sqft Lot:The size of the lot in square feet. Larger lots may increase a property’s desirability and value.
- Floors: The number of floors in the property. Properties with multiple floors may offer more living space and appeal.
- Waterfront: A binary indicator (1 if the property has a waterfront view, 0 other- wise). Properties with waterfront views are often valued higher.
- View: An index from 0 to 4 indicating the quality of the property’s view. Better views are likely to enhance a property’s value.
- Condition: An index from 1 to 5 rating the condition of the property. Properties in better condition are typically worth more.
- Sqft Above: The square footage of the property above the basement. This can help isolate the value contribution of above-ground space.
- Sqft Basement: The square footage of the basement. Basements may add value depending on their usability.
- Yr Built: The year the property was built. Older properties may have historical value, while newer ones may offer modern amenities.
- Yr Renovated: The year the property was last renovated. Recent renovations can increase a property’s appeal and value.
- Street: The street address of the property. This feature can be used to analyze location-specific price trends.
- City: The city where the property is located. Different cities have distinct market dynamics.
- Statezip: The state and zip code of the property. This feature provides regional context for the property.
- Country: The country where the property is located. While this dataset focuseson properties in Australia, this feature is included for completeness.

### Feature Data:
| #   |Column         |Non-Null Count  |Dtype|  
|---  |------         |--------------  |-----|  
| 0   |date           |4140    |object | 
| 1   |price          |4140    |float64|
| 2   |bedrooms       |4140    |float64|
| 3   |bathrooms      |4140    |float64|
| 4   |sqft_living    |4140    |int64  |
| 5   |sqft_lot       |4140    |int64  |
| 6   |floors         |4140    |float64|
| 7   |waterfront     |4140    |int64  |
| 8   |view           |4140    |int64  |
| 9   |condition      |4140    |int64  |
| 10  |sqft_above     |4140    |int64  |
| 11  |sqft_basement  |4140    |int64  |
| 12  |yr_built       |4140    |int64  |
| 13  |yr_renovated   |4140    |int64  |
| 14  |street         |4140    |object |
| 15  |city           |4140    |object |
| 16  |statezip       |4140    |object |
| 17  |country        |4140    |object |

Dari data tersebut bahwa dataset memiliki jumlah 18 kolom dan 4140 baris.

### Missing Values:
|Missing values dalam dataset:| Value |
|-------------------|-----| 
|date         |    0|
|price        |    0|
|bedrooms     |    0|
|bathrooms    |    0|
|sqft_living  |    0|
|sqft_lot     |    0|
|floors       |    0|
|waterfront   |    0|
|view         |    0|
|condition    |    0|
|sqft_above   |    0|
|sqft_basement|    0|
|yr_built     |    0|
|yr_renovated |    0|
|street       |    0|
|city         |    0|
|statezip     |    0|
|country      |    0|

Banyaknya missing values di dataset menunjukkan bahwa dataset tidak ada data yang hilang atau berjumlah 0 (tidak ada).

## Exploratory Data Analysis (EDA)
### Distribution Visualisation
Distribution Visualisation digunakan untuk memvisualisasikan distribusi beberapa fitur numerik dalam dataset untuk memahami bagaimana data pada kolom yang tersebar. Persebaran tersebut antara lain 'price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built'.

![Distribute_DF](https://github.com/user-attachments/assets/302c253e-8c75-4bc7-aa65-4a16d55717de)


### Numerical Predictor Variables dengan Variabel Target
```
#Visualising numerical predictor variables with Target Variables
train_num = df.select_dtypes(include=['int64','float64'])
# Calculate the number of rows needed for subplots
num_rows = int(np.ceil(len(train_num.columns) / 3))
# Create subplots with the calculated number of rows
fig, axs = plt.subplots(num_rows, 3, figsize=(20, 80))

#adjust horizontal space between plots
fig.subplots_adjust(hspace=0.6)
for i, ax in zip(train_num.columns, axs.flatten()):
    sns.scatterplot(x=i, y='price', hue='price', data=train_num, ax=ax, palette='viridis_r')
    plt.xlabel(i, fontsize=12)
    plt.ylabel('price', fontsize=12)
    ax.set_title('price' + ' - ' + str(i), fontweight='bold', size=20)

plt.show()
```

Memvisualisasikan hubungan antara fitur-fitur numerik dalam dataset dengan variabel target, yaitu 'price'. Hal ini dilakukan dengan membuat scatter plot untuk setiap fitur numerik terhadap 'price'.

![Distribute_Numeric](https://github.com/user-attachments/assets/2a08cb23-c616-41f5-ac01-95b2207aa485)


### Categorical Predictor Variables dengan Variabel Target
```
def draw_boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    plt.xticks(rotation=90)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

melted_df = pd.melt(df, id_vars=['price'], value_vars=sorted(categorical_cols))

plot_grid = sns.FacetGrid(melted_df, col="variable", col_wrap=3, sharex=False, sharey=False, height=5)
plot_grid.map(draw_boxplot, "price", "value")

plt.tight_layout()
plt.show()
```

Memvisualisasikan hubungan antara variabel target, yaitu price, dengan fitur-fitur kategorikal dalam dataset. Visualisasi ini menggunakan box plot untuk menunjukkan distribusi harga (price) pada setiap kategori dalam variabel kategorikal tersebut.

![Distribute_Category](https://github.com/user-attachments/assets/e15c59fb-0a81-4ed4-b26e-f9df79377220)


### Outliers
Memvisualisasikan distribusi fitur numerik dalam dataset menggunakan box plot bertujuan untuk mengidentifikasi outlier di luar batas normal.

![Outliers_BF](https://github.com/user-attachments/assets/26b5efe6-b7c3-4aa4-b4b7-1cfae21f217b)

### Skewness and Kurtosis
Menghitung dan menampilkan skewness dan kurtosis dari kolom 'price' pada DataFrame df. Penting Untuk mengetahui apakah perlu melakukan transformasi data dan menghindari masalah pada model regresi.Untuk hasil tersebut adalah 

- Skewness: 24.763681
- Kurtosis: 1010.078975

Dengan hasil tersebut bahwa Skewness memiliki distribusi harga (price) sangat condong ke kanan dan Kurtosis menunjukkan bahwa ada banyak nilai ekstrem/outlier.

### Distribute Feature Numeric Skewness
Visualisasi fitur-fitur numerik dalam dataset setelah kemungkinan dilakukan transformasi untuk menangani skewness.

![Distribute_Skew](https://github.com/user-attachments/assets/0eea18fd-1178-4798-8598-4f096e82e77e)

### Distribution of Target variable (Price)
Visualisasikan distribusi dari kolom 'price' (harga) dalam dataset. Visualisasi distribusi ini membantu memahami sebaran harga rumah, apakah cenderung berkumpul pada nilai tertentu, atau tersebar luas, serta apakah ada nilai-nilai ekstrem.

![Distribute_TVP](https://github.com/user-attachments/assets/eeee41a7-1d9f-47b8-af25-f564212c9416)

### Correlation Distribution
Visualisasi korelasi antara fitur-fitur numerik dalam dataset menggunakan heatmap.

![Distribute_Matrix](https://github.com/user-attachments/assets/d4d03dfd-660c-458a-bfb6-386c164e1d72)

1. Terdapat korelasi sebesar 87% antara sqft_above dengan sqft_living.
2. Terdapat korelasi sebesar 76% antara sqft_living dengan bathroom.
3. Dan fitur independen yang memiliki korelasi yang baik dengan fitur independen lainnya.
   
## Data Preparation
### Penanganan Fitur Yang Tidak Memberikan Informasi dan Membuat Fitur Baru
Menangani fitur yang tidak memberikan manfaat pada performa model prediksi dan menambahkan fitur baru seperti Price per Square Foot = perbandingan harga berdasarkan ukuran rumah yang berbeda. 

```
# remove street country
df = df.drop(['street', 'country','statezip'], axis=1)

# remove year_sold
df = df.drop(['year_sold'],axis=1)

# Convert sqft_basement to 'Has Basement' or 'No Basement'
df['basement_status'] = df['sqft_basement'].apply(lambda x: 'No_Basement' if x == 0 else 'Has_Basement')
#also remove the sqft_basement after
df = df.drop('sqft_basement', axis=1)

# Price per Square Foot = price comparisons across different home sizes.
df['price_per_sqft'] = df['price'] / df['sqft_living']
```

Dengan penangan tersebut dapat membantu dalam menambah informasi yang bermanfaat dalam pelatihan.

### Penanganan Skewness
Penanganan skewness dapat dilakukan apabila fitur numerik yang sebelumnya telah diidentifikasi memiliki kemencengan tinggi dan mentransformasikan fitur-fitur yang memiliki skewness menjadi lebih simetris dan mendekati distribusi normal (Gaussian). Dengan menggunakan transformasi Yeo-Johnson agar distribusinya mendekati normal.

![Distribute_Skew_Norm](https://github.com/user-attachments/assets/57d3e36e-4629-4ea5-9cd3-2888b582c147)

### Penanganan Outliers
Outliers dapat memengaruhi performa model prediksi secara signifikan. Dengan penanganan outlier yang tepat dapat meningkatkan akurasi dan stabilitas performa model. Untuk penanganan tersebut menggunakan  Interquartile Range (IQR).

![Outliers_IQR](https://github.com/user-attachments/assets/23c701a9-78b5-43da-aaaf-ebc299cca0a8)


### One-Hot Encoding
Fitur-fitur kategorikal diubah menjadi format numerik dengan menggunakan one-hot encoding. Dengan mengubah kolom kategorikal menjadi numerikal bertujuan agar data tersebut dapat dengan mudah diolah oleh algoritma machine learning.

### Cross Validation
Mengevaluasi kinerja model secara lebih akurat dan menghindari overfitting atau underfitting dan membantu dalam pemilihan model atau parameter terbaik 

```
kfold= KFold(n_splits=11,random_state=42,shuffle=True) #kfold cross validation

# Error function to compute error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#Assigning scoring paramter to 'neg_mean_squared_error' beacause 'mean_squared_error' is not
# available inside cross_val_score method
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfold))
    return (rmse)
```

###  Splitting data into Trainand Test
Membagi dataset sebesar 70% Training dan 30% Test untuk digunakan dalam pelatihan model.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
```

### SimpleImputer
Mengisi nilai NaN (kosong) dalam dataset dengan rata-rata kolom menggunakan SimpleImputer dari Scikit-Learn.
```
# Menggunakan SimpleImputer untuk mengisi NaN dengan rata-rata kolom
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

### RobustScaler
Mengurangi nilai median dari data dan menskalakan berdasarkan rentang IQR. Supaya model tidak terpengaruh oleh nilai ekstrem pada saat melatih dan menguji data.
```
scaler = RobustScaler()

# Mengubah data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan yaitu Ridge Regression, Lasso Regression, ElasticNet Regression, dan Support Vector Regression (SVR). Dan juga hyperparameter tuning untuk meningkatkan kinerja model.

1. Ridge Regression

    Model Ridge Regression menggunakan pendekatan regularisasi L2 untuk menurunkan kompleksitas model, terutama saat data memiliki multikolinearitas.

    - Konfigurasi: Nilai alpha diseleksi menggunakan GridSearchCV dengan rentang nilai yang cukup detail, mulai dari 5 hingga 15, termasuk nilai desimal kecil seperti 10.35 dan 10.36.

    - Kelebihan: Cenderung menjaga semua fitur tetap dalam model dan sangat bermanfaat ketika fitur saling berkorelasi.

    - Kekurangan: Jika tingkat regularisasi terlalu tinggi, model dapat kehilangan kemampuan dalam menangkap pola penting karena prediksi menjadi terlalu konservatif.

2. Lasso Regression

    Lasso Regression merupakan metode regresi yang memberikan penalti L1, yang secara otomatis dapat mengecilkan beberapa koefisien menjadi nol — sehingga juga berfungsi sebagai alat seleksi fitur.

    - Konfigurasi: Dalam eksperimen ini, alpha ditetapkan pada 0.001 untuk menghindari penalti yang terlalu berat sambil tetap mengontrol overfitting.

    - Kelebihan: Efisien dalam menyaring fitur-fitur yang paling berpengaruh dan mengurangi kompleksitas model.

    - Kekurangan: Dapat mengabaikan fitur dengan kontribusi kecil namun tetap berarti, terutama jika regularisasi tidak disesuaikan secara tepat.

3. ElasticNet Regression

    ElasticNet mengombinasikan regularisasi L1 dan L2, menawarkan fleksibilitas tambahan dengan dua parameter yang bisa disesuaikan.

    - Konfigurasi: Model ini dibangun menggunakan alpha = 0.001 dan l1_ratio = 0.5, sebuah kompromi antara kekuatan seleksi Lasso dan kestabilan Ridge.

    - Kelebihan: Memadukan dua pendekatan regularisasi memungkinkan ElasticNet bekerja baik pada dataset yang memiliki fitur dalam jumlah banyak dan saling berkorelasi.

    - Kekurangan: Membutuhkan pencarian kombinasi dua parameter, sehingga proses tuning bisa lebih rumit dibandingkan model linear biasa.

4. Support Vector Regression (SVR)

    SVR adalah teknik regresi yang mengandalkan konsep margin dan kernel untuk menangani data dengan hubungan yang tidak linear.

    - Konfigurasi: Model disesuaikan dengan C = 19, epsilon = 0.008, dan gamma = 0.00015. Ketiga parameter ini mempengaruhi seberapa ketat margin error dan kompleksitas kurva prediksi.

    - Kelebihan: Cocok untuk memodelkan pola non-linear dan lebih tahan terhadap gangguan dari outlier.

    - Kekurangan: Performa komputasi bisa menurun pada dataset besar dan proses pemilihan parameter lebih menantang.
      
### Improvement Process
- Menggunakan GridSearchCV pada model Ridge Regression dalam menentukan nilai alpha terbaik. Pendekatan ini bertujuan untuk memperoleh tingkat regularisasi yang optimal, sehingga model mampu menghindari overfitting sekaligus tetap menjaga kualitas prediksi secara akurat.

### Model Terbaik pada Solution Statement
Model yang memberikan hasil terbaik yaitu ElasticNet. ElasticNet menggabungkan keunggulan dari Ridge dan Lasso Regression — yaitu kemampuan menangani multikolinearitas dan melakukan seleksi fitur secara otomatis. Hal ini membuatnya lebih fleksibel dalam menghadapi data dengan banyak fitur, baik yang saling berkorelasi maupun yang kurang relevan.

## Evaluation
Projek ini berfokus pada prediksi harga rumah di USA. Tujuan dari proyek ini adalah mengembangkan model machine learning yang mampu memberikan prediksi harga properti secara akurat dan adaptif. Dengan memanfaatkan teknik regularisasi, validasi silang, dan seleksi fitur otomatis, model ini tidak hanya menangani kompleksitas data, tetapi juga memberikan nilai tambah bagi bisnis dalam pengambilan keputusan strategis terkait investasi, penjualan, maupun pembelian properti.

Model yang digunakan dalam projek ini diantaranya Ridge Regression, Lasso Regression, ElasticNet Regression, dan Support Vector Regressor (SVR) dan Evaluasi model menggunakan 2 metrik yaitu MSE (Mean Squared Error) dan RMSE (Root Mean Squared Error).

Evaluasi Model dilakukan dengan menggunakan 2 metrik yaitu 
1. MSE (Mean Squared Error) adalah metrik evaluasi yang digunakan untuk mengukur rata-rata kuadrat selisih antara nilai yang diprediksi oleh model dan nilai aktual
2. RMSE (Root Mean Squared Error) - Digunakan untuk mengukur seberapa besar kesalahan rata-rata antara nilai yang diprediksi dan nilai aktual, dengan mengembalikannya ke satuan asli dari target.

### Hasil Evaluation 

|Model| MSE Train | MSE Test | RMSE Train | RMSE Test |
|---  |------ |--------------  |-----|  -----|  
|Ridge Regression| 0.028491690649740026 | 0.026370277029214496| 0.16879481819576106| 0.16238927621371585|
|Lasso Regression|0.02923064676780914| 0.026268328569776043| 0.17096972471115798| 0.16207507078442399|
|Support Vector Regressor| 0.031025372850044967|0.029611578553164767| 0.176140207931196| 0.17208015153748782|
|ElasticNet Regression| 0.028707714575763144| 0.026049534274832146| 0.1694335107815545| 0.16139868114341005|

### Visualisasi Evaluation 

![Perbandinngan_Model](https://github.com/user-attachments/assets/bca72b90-bb91-4f92-9f66-0a6412960a94)

### Penjelasan Formula Metrik yang Digunakan
Metrik yang digunakan dalam projek ini adalah Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE). Metrik ini dipilih untuk menyelesaikan prediksi regresi.

1. Mean Squared Error (MSE)

    Menghitung rata rata kuadrat antara nilai prediksi ŷᵢ dan nilai aktual yᵢ. Formula MSE adalah

    ![MSE](https://github.com/user-attachments/assets/ebeb7e5c-0a7b-433d-9c71-ae2a55c8635b)

    - ŷᵢ adalah nilai prediksi
    - yᵢ adalah nilai aktual/sebenarnya
    - n adalah jumlah data

    Karena menggunakan kuadrat selisih, MSE memberi penalti besar untuk kesalahan prediksi yang jauh dari nilai aktual.
   
2. Root Mean Squared Error (RMSE)

    RMSE adalah akar dari MSE, yaitu mengukur kesalahan rata-rata antara nilai yang diprediksi dan nilai aktual, dengan mengembalikannya ke satuan asli dari target.

    ![RSME](https://github.com/user-attachments/assets/a5f48e4a-3df7-4dd6-9144-f0611571dbba)

    - ŷᵢ adalah nilai prediksi
    - yᵢ adalah nilai aktual/sebenarnya
    - n adalah jumlah data

    RMSE bersifat lebih representatif karena menyajikan kesalahan prediksi dalam skala yang sama dengan variabel target.

### Hasil Evaluation
1. ElasticNet Regression dan Lasso Regression menunjukkan performa yang akurat, dengan nilai RMSE Test sebesar 0.269, menandakan kemampuan keduanya dalam menghasilkan prediksi yang konsisten terhadap data uji.
2. Ridge Regression juga menunjukkan performa yang sebanding, dengan nilai RMSE Test sebesar 0.271, yang hanya sedikit lebih tinggi dibandingkan ElasticNet dan Lasso, sehingga tetap menjadi model yang andal.
3. Berdasarkan nilai Mean Squared Error (MSE). ElasticNet, Lasso, dan Ridge memiliki nilai MSE Test yang relatif rendah dan konsisten.

### Apakah sudah menjawab setiap problem statment?
Setiap problem statement yang diajukan telah ditanggapi dengan solusi dan pendekatan yang sesuai dalam proses modeling, data preparation, serta evaluasi model. 
- Dalam memprediksi harga rumah yang mampu mengakomodasi berbagai variabel kompleks dapat menggunakan  berbagai algoritma machine learning seperti Ridge, Lasso, ElasticNet, dan SVR yang masing-masing dirancang untuk menangani data dengan variabel kompleks. Dan Visualisasi distribusi, korelasi, dan transformasi data untuk mendukung akurasi prediksi.
- Kemudian Permasalahan multikolinearitas dan overfitting dapat diselesaikan dengan Penggunaan cross-validation untuk mengurangi risiko overfitting. Dan Skewness handling dan outlier removal juga berkontribusi dalam mengatasi noise dan ekstrem data.
- Lalu pemilihan fitur yang relevan dalam membantu memprediksi harga hunian dengan pembuatan fitur baru seperti price_per_sqft dan konversi fitur basement_status. Penghapusan fitur tidak relevan seperti street, country, dan year_sold. Dan Penggunaan Lasso Regression yang secara otomatis melakukan feature selection.

### Apakah berhasil mencapai setiap goals yang diharapkan?

Goal yang diharapkan adalah menggunakan model machine learning yang mampu mengolah berbagai fitur kompleks, mengurangi overfitting dan multikolinearitas dengan regularisasi dan validasi silang, Pemilihan fitur yang relevan untuk meningkatkan akurasi model. Dengan hasil evaluasi menunjukkan bahwa
- Dengan menggunakan empat model regresi: Ridge, Lasso, ElasticNet, dan SVR mampu menangkap pola harga dari berbagai fitur numerik dan kategorikal dengan performa baik.
- Regularisasi dan cross-validation memperkuat generalisasi model dan mengurangi kompleksitas berlebih.
- Menghapus fitur non-informatif (street, country, dll).
- Lasso Regression digunakan sebagai model yang sekaligus melakukan feature selection.

### Apakah setiap solusi statement yang kamu rencanakan berdampak? Jelaskan!
Solusi statement yang direncanakan memiliki dampak yang positif. Dalam evaluasi ini:
- ElasticNet menjadi model terbaik, menunjukkan kombinasi regularisasi L1 dan L2 memberikan hasil paling stabil dan akurat. Lasso Regression membantu melakukan seleksi fitur otomatis. SVR memberikan pendekatan berbeda untuk pola non-linear.
- GridSearchCV berhasil menemukan nilai alpha terbaik untuk Ridge dan Lasso, serta kombinasi C, epsilon, gamma pada SVR. Lalu Model dengan parameter yang sudah dituning menghasilkan RMSE yang lebih rendah, dibandingkan jika menggunakan default parameter.
- Dan juga membantu dalam menentukan model terbaik dengan pendekatan evaluasi yang kuat dan relevan secara bisnis.

Semua solusi yang terapkan berdampak langsung dan positif terhadap proses pemodelan, peningkatan performa, dan pencapaian tujuan proyek.

### Kesimpulan
Proyek ini berhasil membangun model prediktif harga rumah yang akurat dan andal dengan memanfaatkan algoritma machine learning seperti Ridge Regression, Lasso Regression, ElasticNet, dan Support Vector Regression (SVR). Dalam prosesnya, berbagai tahapan penting telah dilakukan, mulai dari eksplorasi data, penanganan skewness dan outlier, transformasi fitur, hingga hyperparameter tuning menggunakan GridSearchCV.
