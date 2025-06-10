# Laporan Proyek Machine Learning - Lutfi Aundrie Hermawan

## Project Overview

Proyek Machine Learning dengan topik sistem rekomendasi ini bertujuan untuk mengembangkan sistem rekomendasi video untuk Amazon Prime Video. Sistem ini dirancang untuk membantu pengguna menemukan film dan serial yang sesuai dengan preferensi mereka secara cepat, efisien, dan personal.

Dengan jumlah konten yang sangat banyak, pengguna sering kali kewalahan dan mengalami kesulitan dalam menemukan tontonan yang sesuai dengan selera mereka. Hal ini dapat menurunkan kualitas pengalaman pengguna dan menyebabkan berkurangnya engagement terhadap platform. Tujuan utama proyek ini adalah untuk meningkatkan pengalaman pengguna (user experience) melalui sistem rekomendasi yang mampu mengurangi waktu pencarian konten dan meningkatkan engagement pengguna terhadap platform.

Untuk mencapai tujuan tersebut, proyek ini akan mengimplementasikan dua pendekatan utama dalam sistem rekomendasi, yaitu:

- Content-Based Filtering – memberikan rekomendasi berdasarkan kesamaan antara konten yang telah disukai pengguna sebelumnya dan konten yang tersedia.

- Collaborative Filtering – memberikan rekomendasi berdasarkan pola perilaku dan preferensi pengguna lain yang memiliki kesamaan dengan pengguna target.

Sistem rekomendasi telah terbukti menjadi elemen penting dalam menciptakan pengalaman pengguna yang lebih personal dan efisien, terutama di platform streaming seperti Amazon Prime Video. Dengan jumlah konten yang terus bertambah, sistem rekomendasi membantu pengguna menemukan film dan serial yang sesuai dengan preferensi mereka tanpa harus mencari secara manual. Amazon menggabungkan pendekatan content-based filtering dan collaborative filtering dalam sistem rekomendasinya, serta mengembangkan algoritma berbasis deep learning untuk meningkatkan relevansi hasil rekomendasi. Menurut laporan dari tim ilmuwan data Amazon, sistem ini telah memberikan kontribusi signifikan dalam meningkatkan engagement dan kepuasan pengguna terhadap platform.

Referensi : [Smith, J., & Linden, G. (2017). The history of Amazon’s recommendation algorithm. Amazon Science.](https://www.amazon.science/the-history-of-amazons-recommendation-algorithm)

## Business Understanding

Amazon Prime Video merupakan salah satu platform streaming terkemuka yang menawarkan lebih dari 20.000 film dan 2.700+ acara TV, mencakup berbagai genre dan kategori usia, mulai dari anak-anak hingga dewasa. Namun, dengan jumlah konten yang begitu banyak, pengguna sering kali merasa kebingungan dalam memilih tontonan yang sesuai dengan preferensi mereka.

Baik pengguna baru maupun lama sering kali tidak memiliki waktu cukup untuk mencari atau menyaring film/acara TV yang sesuai dengan selera mereka, yang akhirnya berdampak pada menurunnya engagement pengguna terhadap platform.

### Problem Statements

Menjelaskan pernyataan masalah:
- Dengan berbagai pilihan tontonan yang menarik di amazon prime, pengguna merasa kesulitan dalam memilih yang mereka suka dan mengakibatkan pengguna mengurangi interaksi dalam memilih tontonan di platform Amazon Prime
- Platform Amazon Prime memiliki banyak tontonan yang tersedia, namun pengguna bingung dalam memilih jenis tontonan atau konten yang sesuai dengan preferensi mereka.
- Meskipun tersedia fitur pencarian, sistem saat ini belum sepenuhnya mampu untuk menyarankan konten baru yang belum pernah dilihat, tetapi berpotensi diminati.

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Mengembangkan sistem rekomendasi yang dapat menyarankan film atau serial secara instan berdasarkan preferensi pengguna. 
- Memberikan rekomendasi tontonan yang lebih personal dan relevan sehingga pengguna lebih terlibat dan puas dengan pengalaman mereka di platform.
- Mengurangi waktu eksplorasi konten dengan memberikan saran yang relevan, bahkan bagi pengguna baru.


### Solution Approach
Untuk mengatasi masalah tersebut, proyek ini akan mengimplementasikan dua pendekatan utama dalam sistem rekomendasi yaitu Content-based Filtering dan Collaborative Filtering.

#### Solution statements
    
- Content-based Filtering

Pendekatan ini merekomendasikan tontonan berdasarkan atribut atau fitur konten yang disukai pengguna sebelumnya. Misalnya Genre film, Aktor atau sutradara, Sinopsis atau deskripsi konten, Rating dan tahun rilis

        
- Collaborative Filtering
  
Pendekatan ini menggunakan data interaksi pengguna, seperti penilaian (rating), tontonan sebelumnya, dan perilaku serupa dengan pengguna lain.
        

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Amazon Prime Movies and TV Shows dataset yang diperoleh dari platform Kaggle. Dataset ini berisi informasi mengenai daftar semua film dan acara TV yang tersedia di Amazon Prime. Dataset tersebut tersedia secara publik dan dapat diunduh melalui Kaggle [Amazon Prime Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows).

Variabel-variabel pada Amazon Prime Movies and TV Shows dataset adalah sebagai berikut:
- show_id : Unique Show ID
- type : Movie or Tv Show
- title : Title of Movie or Show
- director : Director of Movie or Show
- cast : Cast of Movie / Show
- country : Country of Production
- date_added : Date added on Prime
- release_year : Release Year of the movie or show
- rating : Rating of the movie or show
- duration : Duration of the movie or show
- listed_in : Genre
- description : Description of the movie or show

### Feature Data:
| #   |Column         |Non-Null Count  |Dtype|  
|---  |------         |--------------  |-----| 
| 0   |show_id       |9668 non-null  | object|
| 1   |type         | 9668 non-null  | object|
| 2   |title        | 9668 non-null |  object|
| 3   |director     | 7585 non-null |  object|
| 4  | cast        |  8435 non-null |  object|
| 5  | country     |  672 non-null  |  object|
| 6   |date_added  |  155 non-null  |  object|
| 7  | release_year  |9668 non-null|   int64 |
| 8   |rating        |9331 non-null |  object|
| 9  | duration    |  9668 non-null  | object|
| 10 | listed_in    | 9668 non-null  | object|
| 11 | description  | 9668 non-null  | object|

### Missing Values: 

|Missing values dalam dataset:| Value |
|-------------------|-----| 
|show_id         |   0|
|type          |     0|
|title        |      0|
|director     |   2083|
|cast          |  1233|
|country       |  8996|
|date_added   |   9513|
|release_year |      0|
|rating       |    337|
|duration     |      0|
|listed_in    |      0|
|description   |     0|


## Exploratory Data Analysis (EDA):
### Tahun Rilis Terbanyak
Visualisasi distribusi tahun rilis untuk Movie dan TV Show menunjukkan bahwa penambahan konten baru mengalami peningkatan signifikan pada rentang tahun 2015 hingga 2020. Pada periode ini, jumlah konten yang dirilis merupakan yang terbanyak dibandingkan tahun-tahun sebelumnya.
![TahunRilis](https://github.com/user-attachments/assets/efa0798c-f683-4440-be32-ca18793ae628)

### Distribusi Rating Konten
Distribusi konten berdasarkan rating menunjukkan bahwa Amazon Prime memiliki banyak konten yang ditujukan untuk audiens remaja. Hal ini terlihat dari dominasi kategori rating seperti "13+" dan "16+", yang merupakan dua kategori dengan jumlah konten terbanyak di platform tersebut.
![DistribusiRating](https://github.com/user-attachments/assets/e7637192-24f5-45d4-8157-9bae438fcc95)

### Top Genre Amazon Prime
Menvisualisasi genre top 5 di Amazon Prime. Visualisasi menunjukkan bahwa genre yang paling banyak ditampilkan dalam dataset adalah Drama, Comedy, Action, Suspense, dan Kids. Kelima genre ini merupakan genre terpopuler dengan jumlah konten terbanyak.
![TopGenre](https://github.com/user-attachments/assets/b500ce2b-c4a4-44eb-8097-fa228c1fafc5)

### Distribusi Durasi Movie
Visualisasi menunjukkan distribusi durasi film di Amazon Prime, dengan mayoritas Movie memiliki durasi antara 80 hingga 120 menit. Hal ini mencerminkan preferensi umum terhadap film berdurasi standar.
![DistribusiDurasi](https://github.com/user-attachments/assets/a8e73098-4828-4bf8-ac02-ed6b43872836)

### Distribusi Durasi TV Series
Visualisasi menunjukkan distribusi jumlah season pada TV Show di Amazon Prime. Mayoritas serial memiliki hanya 1 season, menunjukkan dominasi format mini series atau limited series di platform ini.
![DistribusiDurasiTV](https://github.com/user-attachments/assets/cab4823b-8cc6-43fb-82ba-3d05d22da0b4)


## Data Preparation
### Handling Missing Values
Penanganan nilai hilang dilakukan untuk menjaga kualitas data dan menghindari bias pada model. Berikut pendekatan yang digunakan:

- Kolom director dan cast diisi dengan nilai "unknown".
- Kolom rating diisi dengan modus (nilai yang paling sering muncul).

Dengan penanganan missing values dapat memberikan peningkatan terhadap kinerja model dan data tidak menjadi bias.

Data sebelum di perbaiki

|Missing values dalam dataset:| Value |
|-------------------|-----| 
|show_id         |   0|
|type          |     0|
|title        |      0|
|director     |   2083|
|cast          |  1233|
|country       |  8996|
|date_added   |   9513|
|release_year |      0|
|rating       |    337|
|duration     |      0|
|listed_in    |      0|
|description   |     0|


Data setelah di perbaiki

|Missing values dalam dataset:| Value |
|-------------------|-----| 
|show_id        |   0|
|type           |   0|
|title           |  0|
|director       |   0|
|cast           |   0|
|release_year   |   0|
|rating         |   0|
|duration        |  0|
|listed_in      |   0|
|description    |   0|

Bedasarkan data missing sebelum diperbaiki terdapat nilai missing values yang tinggi, yaitu 'country' dan 'date_added'. Dari kedua kolom tersebut tidak digunakan karena proporsi nilai hilangnya sangat tinggi dan dinilai tidak relevan terhadap proses rekomendasi.

### Replace Rating
Dalam dataset, ada beberapa nilai yang sebenarnya sama, tetapi ditulis dengan nama atau format yang berbeda. Untuk membuat data lebih konsisten dan mudah dianalisis, nama-nama tersebut diganti atau diseragamkan menjadi satu format yang sama.

```
#basically we are replacing the names to make it less cluster
df['rating']=df['rating'].replace({
    "16": "16+",
    "AGES_16_": "16+",
    "AGES_18_": "18+",
    "R": "18+",
    "NC-17": "18+",
    "13+": "PG-13",
    "PG-13": "PG-13",
    "G": "GENERAL",
    "ALL": "GENERAL",
    "ALL_AGES": "GENERAL",
    "UNRATED": "UNRATED",
    "NOT_RATE": "UNRATED",
    "NR": "UNRATED",
    "7+": "TV-Y7",
    "TV-NR": "TV-UNRATED"    
})
```

### Mengkategorikan Duration
Mengkategorikan duration bertujuan untuk memisahkan antara durasi movie dengan TV series dikarenakan ada perbedaan jenis durasi. 

Durasi diklasifikasikan ke dalam dua bentuk:

- `duration_in_min` untuk Movie, dikonversi dari satuan waktu.
- `duration_in_seasons` untuk TV Show, dihitung dari jumlah season.

```
def condi(x):
    if 'min' in x:
        return int(x.split()[0])
    else:
        return 0
df['duration_in_min']=df['duration'].apply(condi)

df.head()
```

Mendefinisikan duration by season
```
def condi(x):
    if 'season' in x:
        return int(x.split()[0])
    else:
        return 0
df['duration_in_seasons']=df['duration'].apply(condi)

df.head()
```

### Feature Engineering
Menggabungkan berbagai kolom teks (seperti title, director, cast, listed_in, dan description) menjadi satu kolom baru content. Hal ini dilakukan untuk memberikan representasi teks yang lebih komprehensif tentang film atau acara yang ada. Dan membuat fitur gabungan untuk pendekatan Content-Based Filtering

```
# Ambil kolom yang relevan
df_CBF['combined_features'] = df_CBF['title'].fillna('') + ' ' + \
                          df_CBF['director'].fillna('') + ' ' + \
                          df_CBF['cast'].fillna('') + ' ' + \
                          df_CBF['listed_in'].fillna('') + ' ' + \
                          df_CBF['description'].fillna('')
```

### Transformasi Teks ke Vektor Numerik menggunakan TF-IDF Vectorizer

```
# Ubah ke dalam bentuk vektor TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_CBF['combined_features'])

# Cek hasil vektorisasi
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
```

## Modeling
### 1. Model Development Content Based Filtering
Digunakan untuk merekomendasikan film atau TV show berdasarkan kemiripan konten seperti genre, deskripsi, sutradara, dll.

- Parameter : 
-- Vectorizer: `TfidfVectorizer(stop_words='english')`
-- Similarity Measure: `cosine_similarity(tfidf_matrix)`
    
- Kelebihan :
-- Personal dan tidak memerlukan data pengguna lain.
-- Bekerja baik untuk pengguna baru.
  
- Kekurangan :
-- Rentan terhadap rekomendasi monoton (konten terlalu mirip).
-- Bergantung pada kualitas metadata.

### 1. Model Development Collaborative Filtering
Digunakan untuk merekomendasikan film atau TV show berdasarkan kemiripan konten seperti genre, deskripsi, sutradara, dll. Dengan menggunakan pendekatan matrix factorization melalui embedding layer dan neural network sederhana.

- Parameter :
-- Embedding(input_dim=num_users, output_dim=50)
-- Flatten()
-- Concatenate()
-- Dense(64, activation='relu')
-- Dense(1)

- Parameter Pelatihan:
-- Optimizer: Adam
-- Loss: Mean Squared Error
-- Epochs: 40
-- Batch Size: 64

- Kelebihan :
-- Mampu menangkap preferensi kompleks antar pengguna.
-- Memberikan rekomendasi yang beragam.
      
- Kekurangan :
-- Tidak bekerja optimal untuk pengguna/item baru (cold-start).
-- Membutuhkan volume data besar dan preprocessing tambahan.


    
### Improvement Process
- Collaborative Filtering mampu menyediakan rekomendasi yang lebih variatif dan personal dengan memanfaatkan pola interaksi pengguna.
- Content-Based Filtering memberikan rekomendasi yang konsisten bagi pengguna dengan riwayat preferensi yang jelas, namun cenderung menghasilkan rekomendasi yang kurang beragam karena fokus pada konten serupa


### Model Terbaik pada Solution Statement
Bedasarkan hasil modeling bahwa Collaborative Filtering lebih unggul untuk digunakan sebagai model utama karena mampu memberikan prediksi yang lebih akurat terhadap preferensi pengguna. Sementara itu, Content-Based Filtering tetap dapat dimanfaatkan sebagai pelengkap, khususnya untuk menangani kasus cold-start (pengguna baru) atau ketika data interaksi pengguna masih terbatas.

## Evaluation
Projek ini berfokus pada model Content Based Filtering dan model Collaborative Filtering. Tujuan dari proyek ini adalah mengembangkan model Machine Learning yang mampu memberikan referensi movie maupun TV series sesuai dengan referensi pengguna. Dengan menggunakan pendekatan utama dalam sistem rekomendasi dapat memberikan hasil yang baik untuk pengguna.

### Model Content-based Filtering
Untuk proyek sistem rekomendasi ini, metrik evaluasi yang digunakan adalah Precision@K, Recall@K, F1-Score@K.

```
        print(f"\nEvaluation for: '{query_title}'")
        print(f"Top-{k} Recommendations: {recommended_titles[:k]}")
        print(f"Ground Truth: {ground_truth}")
        print(f"\nPrecision@{k}: {precision}")
        print(f"Recall@{k}:    {recall}")
        print(f"F1-Score@{k}:  {f1}")
```
```
evaluate_recommendation_system("The Cat in the Hat Knows a Lot About Halloween!", relevant_movies, k=10)
```

Evaluation for: 'The Cat in the Hat Knows a Lot About Halloween!'
Top-10 Recommendations: ['The Cat in the Hat Knows a Lot About Christmas', 'The Cat in the Hat Knows a Lot About Camping!', 'The Cat in the Hat Knows a Lot About Space!', 'The Cat in the Hat Knows a Lot About That!', 'Morphle Halloween Special - The Halloween Candy Magic Pet', 'CoComelon Halloween Songs', 'Oddbods - Halloween Special', 'Steve and Maggie - Haunted Halloween Special (Vol. 4)', 'Rhymes for Kids and Babies - Spooky Halloween Songs - Mother Goose Club', 'Halloween Kids Songs by Little Baby Bum']
Ground Truth: ['Faster', 'Halloween Kids Songs by Little Baby Bum', 'Halloween Heroes', '27 September']

Precision@10: 0.1
Recall@10:    0.25
F1-Score@10:  0.1429
{'precision': 0.1, 'recall': 0.25, 'f1_score': 0.1429}


#### Penjelasan Metrik yang digunakan
Metrik Evaluasi: Precision, Recall, dan F1-Score
1. Precision
   Precision mengukur seberapa banyak item yang direkomendasikan yang relevan dibandingkan dengan jumlah total item yang direkomendasikan. Formula Precision adalah

   ![Precision](https://github.com/user-attachments/assets/01506070-84f8-4c43-8281-52c198509f68)

Precision yang tinggi berarti sistem jarang memberikan rekomendasi yang tidak relevan. Cocok digunakan ketika kualitas hasil lebih penting daripada kuantitas.

2. Recall
   Recall mengukur seberapa banyak item relevan yang direkomendasikan dibandingkan dengan total item relevan yang ada. Formula Recall adalah

   ![Recall](https://github.com/user-attachments/assets/3e186415-da97-4fe1-b1d8-76cbc68ded3f)

Recall yang tinggi menunjukkan bahwa sistem mampu menangkap sebagian besar item relevan dalam daftar rekomendasinya.

3. F1-Score
   F1-Score adalah rata-rata harmonik dari precision dan recall. Digunakan ketika Anda ingin menyeimbangkan antara precision dan recall. Formula F1-Score adalah

![f1-score_1](https://github.com/user-attachments/assets/73996494-09ba-4881-8946-d30e0ef9583c)

F1-Score yang tinggi menunjukkan bahwa sistem tidak hanya memberikan rekomendasi yang relevan (precision), tetapi juga mencakup sebagian besar item relevan yang tersedia (recall).   

#### Hasil Metrik Content-based Filtering
- Precision 0.1 berarti 10% dari rekomendasi tepat sasaran.
- Recall 0.25 menunjukkan 25% konten relevan berhasil ditangkap.
- F1-Score 0.1429 mencerminkan keseimbangan yang belum optimal antara precision dan recall.



### Model Collaborative Filtering
Untuk proyek sistem rekomendasi ini, metrik evaluasi yang digunakan adalah Root Mean Squared Error (RMSE) dan Mean Squared Error (MSE). Kedua metrik ini digunakan untuk mengukur akurasi prediksi rating film terhadap rating yang sebenarnya.

```
# Mengambil data MSE dan RMSE dari history
mse = history.history['mean_squared_error']
rmse = history.history['rmse']
```

#### Penjelasan Metrik yang digunakan
Metrik yang digunakan dalam projek ini adalah Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE).

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


#### Visualisasi Metrik Collaborative Filtering

![EvaluationCF](https://github.com/user-attachments/assets/2163ed62-b9c2-40ca-b2a8-dd55fcf9763e)

#### Hasil Evaluation Metrics

1. Penurunan MSE dan RMSE menunjukkan bahwa model berhasil mempelajari hubungan antara pengguna dan item (film) dengan cukup baik seiring berjalannya waktu.
2. RMSE yang stabil di sekitar angka 4.2 menunjukkan bahwa model tidak lagi mengalami overfitting atau underfitting secara signifikan setelah melewati sekitar 20 epoch.
3. Namun, meskipun MSE rendah, nilai RMSE yang masih cukup tinggi menunjukkan bahwa masih terdapat deviasi yang lumayan antara prediksi dan nilai aktual—yang bisa jadi disebabkan oleh variasi data pengguna yang kompleks atau sparsitas data.


## Hasil Akhir Project
### Apakah sudah menjawab setiap problem statment?
Setiap problem statement yang diajukan telah ditanggapi dengan solusi dan pendekatan yang sesuai dalam proses modeling, data preparation, serta evaluasi model.

- Pengguna mendapatkan rekomendasi personal tanpa harus mencari manual.
- Mengurangi waktu eksplorasi konten dan meningkatkan engagement pengguna.
- Memberikan alternatif tontonan yang sesuai dengan preferensi pengguna.

### Apakah berhasil mencapai setiap goals yang diharapkan?
Goal yang diharapkan adalah menggunakan model machine learning yang mampu mengolah berbagai fitur kompleks, mengurangi overfitting dan multikolinearitas dengan regularisasi dan validasi silang, Pemilihan fitur yang relevan untuk meningkatkan akurasi model. Dengan hasil evaluasi menunjukkan bahwa

- Menyarankan film/serial secara instan sesuai preferensi pengguna:
Model Content-Based Filtering memberikan rekomendasi langsung berdasarkan metadata konten yang telah disukai.
- Memberikan rekomendasi yang personal dan relevan:
Collaborative Filtering menghasilkan rekomendasi yang bersifat personal dengan mempelajari pola interaksi pengguna terhadap berbagai konten, terbukti dari hasil evaluasi dengan MSE dan RMSE yang menurun secara stabil.
- Mengurangi waktu eksplorasi konten, bahkan bagi pengguna baru:
Penggabungan dua pendekatan (CBF dan CF) memberikan solusi yang juga dapat diterapkan untuk pengguna baru (cold-start), khususnya dengan Content-Based Filtering.

### Apakah setiap solusi statement yang kamu rencanakan berdampak? Jelaskan!
Solusi statement yang direncanakan memiliki dampak yang positif. Dalam evaluasi ini:

- Content-Based Filtering:
-- Memberikan rekomendasi yang relevan dan konsisten berdasarkan konten yang mirip dengan preferensi pengguna.
-- Dampak: Bermanfaat untuk pengguna baru dan memperkuat rekomendasi awal.

- Collaborative Filtering:
-- Menyediakan rekomendasi berdasarkan pola perilaku pengguna lain, sehingga lebih variatif dan adaptif.
-- Dampak: Meningkatkan personalisasi rekomendasi dan memperkaya eksplorasi konten baru.
  
Semua solusi yang terapkan berdampak langsung dan positif terhadap proses pemodelan, peningkatan performa, dan pencapaian tujuan proyek.

### Kesimpulan
Kedua solusi yang diimplementasikan tidak hanya relevan terhadap permasalahan yang diangkat, tetapi juga efektif dalam mencapai tujuan proyek. Penggabungan keduanya (Hybrid Model) direkomendasikan ke depannya untuk meningkatkan kualitas rekomendasi secara menyeluruh baik dari segi relevansi maupun keberagaman.
