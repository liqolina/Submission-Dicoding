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
- Dengan berbagai pilihan tontonan yang menarik di Amazon Prime, pengguna merasa kesulitan dalam memilih yang mereka suka dan mengakibatkan pengguna mengurangi interaksi dalam memilih tontonan di platform Amazon Prime
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
- id: The title ID on JustWatch.
- title: The name of the title.
- show_type: TV show or movie.
- description: A brief description.
- release_year: The release year.
- age_certification: The age certification.
- runtime: The length of the episode (SHOW) or movie.
- genres: A list of genres.
- production_countries: A list of countries that - produced the title.
- seasons: Number of seasons if it's a SHOW.
- imdb_id: The title ID on IMDB.
- imdb_score: Score on IMDB.
- imdb_votes: Votes on IMDB.
- tmdb_popularity: Popularity on TMDB.
- tmdb_score: Score on TMDB.

### Feature Data:
#### Dataset Titles
| #  | Column               |            Non-Null Count | Dtype   |
|----|----------------------|----------------------------|---------|
| 0  | id                   |        10873 non-null     | object  |
| 1  | title                |        10873 non-null     | object  |
| 2  | type                 |        10873 non-null     | object  |
| 3  | description          |        10729 non-null     | object  |
| 4  | release_year         |        10873 non-null     | int64   |
| 5  | age_certification    |         3688 non-null     | object  |
| 6  | runtime              |        10873 non-null     | int64   |
| 7  | genres               |        10873 non-null     | object  |
| 8  | production_countries |        10873 non-null     | object  |
| 9  | seasons              |         1551 non-null     | float64 |
| 10 | imdb_id              |        10172 non-null     | object  |
| 11 | imdb_score           |         9765 non-null     | float64 |
| 12 | imdb_votes           |         9753 non-null     | float64 |
| 13 | tmdb_popularity      |        10302 non-null     | float64 |
| 14 | tmdb_score           |         8747 non-null     | float64 |

#### Dataset Credits
| #   | Column     | Non-Null Count   | Dtype  |
|-----|------------|------------------|--------|
| 0   | person_id  | 140553 non-null  | int64  |
| 1   | id         | 140553 non-null  | object |
| 2   | name       | 140553 non-null  | object |
| 3   | character  | 122705 non-null  | object |
| 4   | role       | 140553 non-null  | object |



### Missing Values: 

|Missing values dalam dataset:| Value |
|-----------|---------------|
| id        | 0             |   
| title     | 0             |   
| type      | 0             |  
| description | 144         |   
| release_year | 0          |   
| age_certification | 7185   |   
| runtime   | 0             |  
| genres    | 0             |   
| production_countries | 0   |   
| seasons   | 9322          |   
| imdb_id   | 701           |   
| imdb_score | 1108         |   
| imdb_votes | 1120         |   
| tmdb_popularity | 571     |   
| tmdb_score | 2126         |   



## Exploratory Data Analysis (EDA):
### Tahun Rilis Terbanyak
Visualisasi distribusi tahun rilis untuk Movie dan TV Show menunjukkan bahwa penambahan konten baru mengalami peningkatan signifikan pada rentang tahun 2015 hingga 2020. Pada periode ini, jumlah konten yang dirilis merupakan yang terbanyak dibandingkan tahun-tahun sebelumnya.
![TahunRilis](https://github.com/user-attachments/assets/3d8d830c-f0f4-4802-90ac-8a9b20ee9a1c)


### Distribusi Age Certification
Distribusi konten berdasarkan Age Certification menunjukkan bahwa Amazon Prime memiliki banyak konten yang ditujukan untuk audiens remaja. Hal ini terlihat dari dominasi kategori Age Certification seperti "R" dan "PG-13", yang merupakan dua kategori dengan jumlah konten terbanyak di platform tersebut.
![DistribusiAgeCertification](https://github.com/user-attachments/assets/3c21a7cd-7fbe-4f47-a881-b3959d0bab77)


### Top Genre Amazon Prime
Menvisualisasi genre top 5 di Amazon Prime. Visualisasi menunjukkan bahwa genre yang paling banyak ditampilkan dalam dataset adalah Drama, Comedy, Action, Suspense, dan Kids. Kelima genre ini merupakan genre terpopuler dengan jumlah konten terbanyak.
![TopGenre](https://github.com/user-attachments/assets/dadc1423-0e74-448d-ad66-367846c288fc)


### Distribusi Durasi Movie
Visualisasi menunjukkan distribusi durasi film di Amazon Prime, dengan mayoritas Movie memiliki durasi antara 80 hingga 120 menit. Hal ini mencerminkan preferensi umum terhadap film berdurasi standar.
![DistribusiDurasi](https://github.com/user-attachments/assets/31a77647-0b61-4aa0-9e1b-0f35d49951ad)


### Distribusi Durasi TV Series
Visualisasi menunjukkan distribusi jumlah season pada TV Show di Amazon Prime. Mayoritas serial memiliki hanya 1 season, menunjukkan dominasi format mini series atau limited series di platform ini.
![DistribusiDurasiTV](https://github.com/user-attachments/assets/5a210e12-c679-4734-81b7-d1447006e2ac)



## Data Preparation
### Handling Missing Values
Penanganan nilai hilang dilakukan untuk menjaga kualitas data dan menghindari bias pada model. Berikut pendekatan yang digunakan:

- Kolom Age Certification akan diisi dengan data "Unrated".
- Kolom seasons akan diisi dengan nilai 0 untuk konten jenis Movie yang memang tidak memiliki season.

Dengan penanganan missing values dapat memberikan peningkatan terhadap kinerja model dan data tidak menjadi bias.

Data sebelum di perbaiki

|Missing values dalam dataset:| Value |
|-----------|---------------|
| id        | 0             |   
| title     | 0             |   
| type      | 0             |  
| description | 144         |   
| release_year | 0          |   
| age_certification | 7185   |   
| runtime   | 0             |  
| genres    | 0             |   
| production_countries | 0   |   
| seasons   | 9322          |   
| imdb_id   | 701           |   
| imdb_score | 1108         |   
| imdb_votes | 1120         |   
| tmdb_popularity | 571     |   
| tmdb_score | 2126         | 


Data setelah di perbaiki

|Missing values dalam dataset:| Value |
|-------------------|-----| 
| id                  | 0     |   
| title               | 0     |  
| type                | 0     |   
| description         | 0     |   
| release_year        | 0     |   
| age_certification   | 0     |   
| runtime             | 0     |  
| genres              | 0     |   
| production_countries| 0     |   
| seasons             | 0     |   
| imdb_id             | 701   |   
| imdb_score          | 0     |   
| imdb_votes          | 0     |   
| tmdb_popularity     | 0     |  
| tmdb_score          | 0     |   
| year_group          | 933   |  
| has_imdb            | 0     |   



### Handling Data
Dalam dataset, ada kolom yang berisi string list. Dengan format tersebut, akan menganggu proses modeling karena adanya simbol dan lainnya.

```
# Pastikan kolom 'genres' dan 'production_countries' adalah string
df['genres'] = df['genres'].astype(str)
df['production_countries'] = df['production_countries'].astype(str)

# Mengekstrak genre pertama dan negara produksi pertama dari kolom yang berisi string list
df['genres'] = df['genres'].str.replace('[', '', regex=False)\
                                   .str.replace(']', '', regex=False)\
                                   .str.replace("'", '', regex=False)
df['genre'] = df['genres'].str.split(',').str[0].str.strip()
```

Memisahkan Role Actor dan Director supaya mempermudah dalam pemprosesan
```
# Pisahkan DIRECTOR
directors = df_credits[df_credits['role'] == 'DIRECTOR'].groupby('id')['name'].first().reset_index()
directors.columns = ['id', 'director']
print(directors.head())

df = df.merge(directors, on='id', how='left')
```


### Feature Engineering
Menggabungkan berbagai kolom teks (seperti title, director, cast, genres, dan description) menjadi satu kolom baru content. Hal ini dilakukan untuk memberikan representasi teks yang lebih komprehensif tentang film atau acara yang ada. Dan membuat fitur gabungan untuk pendekatan Content-Based Filtering

```
# Ambil kolom yang relevan
df_CBF['combined_features'] = df_CBF[['title', 'description', 'cast', 'genres', 'director']] \
                    .fillna('').agg(' '.join, axis=1)
```

### Transformasi Teks ke Vektor Numerik menggunakan TF-IDF Vectorizer
Teknik pemrosesan teks yang digunakan untuk mengukur seberapa penting sebuah kata dalam sebuah dokumen relatif terhadap seluruh koleksi dokumen (corpus).

```
# Ubah ke dalam bentuk vektor TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_CBF['combined_features'])

# Cek hasil vektorisasi
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
```

## Modeling
### 1. Model Development Content Based Filtering
Digunakan untuk merekomendasikan film atau TV show berdasarkan kemiripan konten seperti genre, deskripsi, sutradara, dll. Dengan menggunakan pendekatan matrix factorization melalui embedding layer dan neural network sederhana.

- Parameter : 
-- Vectorizer: `TfidfVectorizer(stop_words='english')`
-- Similarity Measure: `cosine_similarity(tfidf_matrix)`
    
- Kelebihan :
-- Personal dan tidak memerlukan data pengguna lain.
-- Bekerja baik untuk pengguna baru.
  
- Kekurangan :
-- Rentan terhadap rekomendasi monoton (konten terlalu mirip).
-- Bergantung pada kualitas metadata.

### 2. Model Development Collaborative Filtering
Digunakan untuk merekomendasikan film atau TV show berdasarkan kemiripan konten seperti genre, deskripsi, sutradara, dll.

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

Dan tambahan Sistem Rekomendasi Menggunakan Collaborative Filltering
    
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
evaluate_recommendation_system("Bleed", relevant_movies, k=10)
```


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

Evaluation for: 'Bleed'
Top-10 Recommendations: ['DIVE!!', 'Digging to Death', 'Girl, Chill', 'Twinsanity', 'Cave Club', 'Chicago Massacre: Richard Speck', 'Seven Alone', 'Hellblock 13', 'Devil in the Flesh', 'Erik Terrell: Live at the Helium Comedy Club']
Ground Truth: ['Putham Pudhu Kaalai ', 'Digging to Death', 'All Through the House', 'Sita Ramam']

Precision@10: 0.1
Recall@10:    0.25
F1-Score@10:  0.1429
{'precision': 0.1, 'recall': 0.25, 'f1_score': 0.1429}


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
![EvaluationCF](https://github.com/user-attachments/assets/e2221ab5-f8cd-472c-a78b-7efebd9f862a)



#### Hasil Metrik Model Collaborative Filtering

Top 10 rekomendasi film untuk user 70:

1. Title: The simulators  |  Genre: action, comedy
2. Title: Men of the Deeps  |  Genre: documentation
3. Title: Spirit of Love: The Mike Glenn Story  |  Genre: sport, family
4. Title: Unity: The Latin Tribute to Michael Jackson  |  Genre: music
5. Title: Harmony with A. R. Rahman  |  Genre: documentation, music
6. Title: Alexander Babu: Alex in Wonderland  |  Genre: comedy
7. Title: Water Helps the Blood Run  |  Genre: comedy, drama
8. Title: Clarkson's Farm  |  Genre: reality, documentation, comedy
9. Title: Stracci  |  Genre: documentation
10. Title: Because We're Done  |  Genre: comedy

A. Model berhasil belajar dengan cepat dan menyesuaikan diri terhadap data di beberapa epoch pertama.

B. Proses pelatihan menunjukkan konvergensi yang stabil, dengan penurunan error yang konsisten.

C. Nilai RMSE yang tetap di kisaran ~1.8 mengindikasikan bahwa masih ada ruang untuk perbaikan, kemungkinan dari segi arsitektur model, preprocessing data, atau hyperparameter tuning.


## Hasil Akhir Project
### Apakah sudah menjawab setiap problem statment?
Setiap problem statement yang diajukan telah ditanggapi dengan solusi dan pendekatan yang sesuai dalam proses modeling, data preparation, serta evaluasi model.

- Pengguna mendapatkan rekomendasi personal tanpa harus mencari manual.
- Mengurangi waktu eksplorasi konten dan meningkatkan engagement pengguna.
- Memberikan alternatif tontonan yang sesuai dengan preferensi pengguna.

### Apakah berhasil mencapai setiap goals yang diharapkan?
Goal yang diharapkan adalah menggunakan model machine learning yang mampu mengolah berbagai fitur. Pemilihan fitur yang relevan untuk meningkatkan akurasi model. Dengan hasil evaluasi menunjukkan bahwa :

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
