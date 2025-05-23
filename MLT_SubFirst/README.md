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

1. Menggunakan beberapa algoritma seperti Ridge, Lasso, ElasticNet, dan SVR.
2. Meningkatkan akurasi dan performa model dengan Hyperparameter Tuning.
3. Menggunakan metrix evaluasi seperti RMSE dan MSE.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah USA Housing Dataset yang diperoleh dari platform Kaggle. Dataset ini berisi informasi mengenai sejumlah fitur yang mempengaruhi harga rumah di Amerika Serikat. Dataset tersebut tersedia secara publik dan dapat diunduh melalui Kaggle [USA Housing Dataset](https://www.kaggle.com/datasets/fratzcan/usa-house-prices).

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

## Exploratory Data Analysis (EDA)
### Distribution Visualisation
```
# Define key numerical features
features = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']

# Adjust the grid layout dynamically based on the number of features
num_features = len(features)
rows = (num_features // 3) + (num_features % 3 > 0)  # Calculate required rows
cols = 3  # 3 columns for better spacing

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(18, 12))  # Adjust figure size for clarity
axes = axes.flatten()  # Flatten for easier indexing

# Create histograms for each feature
for i, feature in enumerate(features):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color=sns.color_palette("viridis")[i])
    axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(feature, fontsize=11)
    axes[i].set_ylabel('Frequency', fontsize=11)

# Hide any extra empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Optimize layout
plt.tight_layout()
plt.show()
```
![](https://github.com/liqolina/Submission-Dicoding/blob/main/MLT_SubFirst/IMG/Distribute_DF.png)

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
![](https://github.com/liqolina/Submission-Dicoding/blob/main/MLT_SubFirst/IMG/Distribute_Numeric.png)

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
![](https://github.com/liqolina/Submission-Dicoding/blob/main/MLT_SubFirst/IMG/Distribute_Category.png)

### Distribution of Target variable (Price)
```
#Correlation between variables to check multicollinearity
# Generate a mask for the upper triangle (taken from seaborn example gallery)
plt.subplots(figsize = (30,20))
# Changed np.bool to bool
mask = np.zeros_like(train_num.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True
#Plotting heatmap
sns.heatmap(train_num.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0)
```

![](https://github.com/liqolina/Submission-Dicoding/blob/main/MLT_SubFirst/IMG/Distribute_Matrix.png)

1. Terdapat korelasi sebesar 87% antara sqft_above dengan sqft_living.
2. Terdapat korelasi sebesar 76% antara sqft_living dengan bathroom.
3. Dan fitur independen yang memiliki korelasi yang baik dengan fitur independen lainnya.
   
## Data Preparation
### Penanganan Fitur Yang Tidak Memberikan Informasi dan Membuat Fitur Baru
Menangani fitur yang tidak memberikan manfaat pada performa model prediksi dan menambahkan fitur baru seperti Price per Square Foot = perbandingan harga berdasarkan ukuran rumah yang berbeda.

```
# remove street country
df = df.drop(['street', 'country','statezip'], axis=1)

#make the date column into year and month
df['date'] = pd.to_datetime(df['date'])
df['year_sold']= df['date'].dt.year
df['month_sold'] = df['date'].dt.month

#remove the date column now
df = df.drop('date', axis=1)

# remove year_sold
df = df.drop(['year_sold'],axis=1)

# Convert yr_renovated to 'Never Renovated' or 'Renovated'
df['renovation_status'] = df['yr_renovated'].apply(lambda x: 'Never_Renovated' if x == 0 else 'Renovated')
#also remove yr_renovated column
df = df.drop('yr_renovated', axis=1)

# Convert sqft_basement to 'Has Basement' or 'No Basement'
df['basement_status'] = df['sqft_basement'].apply(lambda x: 'No_Basement' if x == 0 else 'Has_Basement')
#also remove the sqft_basement after
df = df.drop('sqft_basement', axis=1)

# Price per Square Foot = price comparisons across different home sizes.
df['price_per_sqft'] = df['price'] / df['sqft_living']
```

### Penanganan Outliers
Outliers dapat menganggu hasil performa model prediksi, perlu dihapus dari data.

```
# Columns for IQR filtering
columns_to_filter = ['price', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement']

# Function to remove outliers using IQR
def remove_outliers_iqr(x, columns):
    for col in columns:
        Q1 = x[col].quantile(0.35)
        Q3 = x[col].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        x = x[(x[col] >= lower_bound) & (x[col] <= upper_bound)]
    return x

# Apply IQR method
df = remove_outliers_iqr(df, columns_to_filter)

# Display the cleaned data
print(df.shape)
```
### Penanganan Skewness
Variabel numerik yang memiliki skewness diatasi dengan menggunakan transformasi Yeo-Johnson agar distribusinya mendekati normal.

```
transformer = PowerTransformer(method='yeo-johnson')
df_all = df
df_all[skew_index] = transformer.fit_transform(df_all[skew_index])
```

### One-Hot Encoding
Fitur-fitur kategorikal diubah menjadi format numerik dengan menggunakan one-hot encoding.

```
df_all_num= df_all.select_dtypes(include=['float64','int64']).columns  # Numerical columns
df_all_temp = df_all.select_dtypes(exclude=['float64','int64']) # selecting object and categorical features only
df_all_dummy= pd.get_dummies(df_all_temp)
df_all=pd.concat([df_all,df_all_dummy],axis=1) # joining converted dummy feature and original df_all dataset
df_all= df_all.drop(df_all_temp.columns,axis=1) #removing original categorical columns
df_all.shape
```

### Cross Validation
Mengevaluasi kinerja model secara lebih akurat dan menghindari overfitting atau underfitting.

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

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
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

Kekurangan: Membutuhkan pencarian kombinasi dua parameter, sehingga proses tuning bisa lebih rumit dibandingkan model linear biasa.

4. Support Vector Regression (SVR)
SVR adalah teknik regresi yang mengandalkan konsep margin dan kernel untuk menangani data dengan hubungan yang tidak linear.

- Konfigurasi: Model disesuaikan dengan C = 19, epsilon = 0.008, dan gamma = 0.00015. Ketiga parameter ini mempengaruhi seberapa ketat margin error dan kompleksitas kurva prediksi.

- Kelebihan: Cocok untuk memodelkan pola non-linear dan lebih tahan terhadap gangguan dari outlier.

- Kelemahan: Performa komputasi bisa menurun pada dataset besar dan proses pemilihan parameter lebih menantang.

## Evaluation
Projek ini berfokus pada prediksi harga rumah di USA. Tujuan dari proyek ini adalah mengembangkan model machine learning yang mampu memberikan prediksi harga properti secara akurat dan adaptif. Dengan memanfaatkan teknik regularisasi, validasi silang, dan seleksi fitur otomatis, model ini tidak hanya menangani kompleksitas data, tetapi juga memberikan nilai tambah bagi bisnis dalam pengambilan keputusan strategis terkait investasi, penjualan, maupun pembelian properti.

Model yang digunakan dalam projek ini diantaranya Ridge Regression, Lasso Regression, ElasticNet Regression, dan Support Vector Regressor (SVR) dan Evaluasi model menggunakan 2 metrik yaitu MSE (Mean Squared Error) dan RMSE (Root Mean Squared Error).

### Tabel Evaluation 

|Model| MSE Train | MSE Test | RMSE Train | RMSE Test |
|---  |------ |--------------  |-----|  -----|  
|Ridge Regression| 0.0443002905053288 | 0.07363158066095815| 0.2104763419135956| 0.27135139701309474|
|Lasso Regression|0.04491836610435136| 0.07253465254922058| 0.21193953407599858| 0.269322580837962|
|Support Vector Regressor| 0.05068377369753542|0.08562694878252852| 0.22513057033094244| 0.292620827663597|
|ElasticNet Regression| 0.044542234401517077| 0.07257195138834753| 0.21105031248855585| 0.2693918175972454|

### Visualisasi Evaluation 
![](https://github.com/liqolina/Submission-Dicoding/blob/main/MLT_SubFirst/IMG/Perbandinngan_Model.png)
