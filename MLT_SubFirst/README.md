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
| Column | Non-Null Count | Dtype |
| ------------- | ------------- |
| date | 4140 non-null | object |
| price | 4140 non-null | float64 |
|bedrooms|4140 non-null|float64|
|bathrooms|4140 non-null|float64|
|sqft_living|4140 non-null|int64| 
|sqft_lot|4140 non-null|int64|  
 6   floors         4140 non-null   float64
 7   waterfront     4140 non-null   int64  
 8   view           4140 non-null   int64  
 9   condition      4140 non-null   int64  
 10  sqft_above     4140 non-null   int64  
 11  sqft_basement  4140 non-null   int64  
 12  yr_built       4140 non-null   int64  
 13  yr_renovated   4140 non-null   int64  
 14  street         4140 non-null   object 
 15  city           4140 non-null   object 
 16  statezip       4140 non-null   object 
 17  country        4140 non-null   object 

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

