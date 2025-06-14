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

### Dataset Information

Variabel-variabel pada Amazon Prime Movies and TV Shows dataset beserta Feature Data adalah sebagai berikut:

#### Dataset Titles
| #  | Feature               | Non-Null Count | Dtype   | Description                                  |
| -- | --------------------- | -------------- | ------- | -------------------------------------------- |
| 0  | id                    | 10873 non-null | object  | The title ID on JustWatch.                   |
| 1  | title                 | 10873 non-null | object  | The name of the title.                       |
| 2  | type                  | 10873 non-null | object  | TV show or movie.                            |
| 3  | description           | 10729 non-null | object  | A brief description.                         |
| 4  | release\_year         | 10873 non-null | int64   | The release year.                            |
| 5  | age\_certification    | 3688 non-null  | object  | The age certification.                       |
| 6  | runtime               | 10873 non-null | int64   | The length of the episode (SHOW) or movie.   |
| 7  | genres                | 10873 non-null | object  | A list of genres.                            |
| 8  | production\_countries | 10873 non-null | object  | A list of countries that produced the title. |
| 9  | seasons               | 1551 non-null  | float64 | Number of seasons if it's a SHOW.            |
| 10 | imdb\_id              | 10172 non-null | object  | The title ID on IMDB.                        |
| 11 | imdb\_score           | 9765 non-null  | float64 | Score on IMDB.                               |
| 12 | imdb\_votes           | 9753 non-null  | float64 | Votes on IMDB.                               |
| 13 | tmdb\_popularity      | 10302 non-null | float64 | Popularity on TMDB.                          |
| 14 | tmdb\_score           | 8747 non-null  | float64 | Score on TMDB.                               |


#### Dataset Credits
| # | Column     | Non-Null Count  | Dtype  | Description                   |
| - | ---------- | --------------- | ------ | ----------------------------- |
| 0 | person\_id | 140553 non-null | int64  | The person ID on JustWatch.   |
| 1 | id         | 140553 non-null | object | The title ID on JustWatch.    |
| 2 | name       | 140553 non-null | object | The actor or director's name. |
| 3 | character  | 122705 non-null | object | The character name.           |
| 4 | role       | 140553 non-null | object | ACTOR or DIRECTOR.            |
  

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
### Model Development Content Based Filtering
#### Missing Values pada Dataset Titles :
1. Mengecek data pada kolom yang memiliki nilai hilang (Missing Values)
   
| **Kolom**             | **Missing Values (Total)** | **Missing Values (%)** |
| --------------------- | -------------------------- | ---------------------- |
| seasons               | 9.322                      | 85,74%                 |
| age\_certification    | 7.185                      | 66,08%                 |
| tmdb\_score           | 2.126                      | 19,55%                 |
| imdb\_votes           | 1.120                      | 10,30%                 |
| imdb\_score           | 1.108                      | 10,19%                 |
| imdb\_id              | 701                        | 6,45%                  |
| tmdb\_popularity      | 571                        | 5,25%                  |
| description           | 144                        | 1,32%                  |
| id                    | 0                          | 0%                     |
| title                 | 0                          | 0%                     |
| type                  | 0                          | 0%                     |
| release\_year         | 0                          | 0%                     |
| runtime               | 0                          | 0%                     |
| genres                | 0                          | 0%                     |
| production\_countries | 0                          | 0%                     |


Berdasarkan data pada dataset, terdapat nilai hilang (missing value) yang cukup tinggi pada kolom seasons dan age_certification, yaitu dengan persentase di atas 60%. Namun untuk "seasons" tidak perlu didrop dikarenakan pada "typre : Movie" tidak ada season dan hanya perlu diisi nilai 0. Sedangkan "age_certification" untuk nilai kosong diisi "UnRated"

2. Handling Missing Values
Penanganan nilai hilang dilakukan untuk menjaga kualitas data dan menghindari bias pada model. Berikut pendekatan yang digunakan:

- Kolom Age Certification akan diisi dengan data "Unrated".
- Kolom seasons akan diisi dengan nilai 0 untuk konten jenis Movie yang memang tidak memiliki season.
- Kolom imdb_score, tmdb_score, dan tmdb_popularity akan diisi dengan nilai tengah.
- Kolom imdb_votes akan diisi dengan nilai 0 karena tidak ada yang menvoting oleh user
- Kolom description akan diisi dengan data "No description available.".

Dengan penanganan missing values dapat memberikan peningkatan terhadap kinerja model dan data tidak menjadi bias.

3. Mengecek kembali data setelah di perbaiki

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



#### Handling Data
- Memaskikan kolom 'genres' dan 'production_countries' dengan format string

```
# Pastikan kolom 'genres' dan 'production_countries' adalah string
df_titles['genres'] = df_titles['genres'].astype(str)
df_titles['production_countries'] = df_titles['production_countries'].astype(str)
```

- Dalam dataset, ada kolom yang berisi string list. Dengan format tersebut, akan menganggu proses modeling karena adanya simbol dan lainnya.

```
# Mengekstrak genre pertama dan negara produksi pertama dari kolom yang berisi string list
df_titles['genres'] = df_titles['genres'].str.replace('[', '', regex=False)\
                                   .str.replace(']', '', regex=False)\
                                   .str.replace("'", '', regex=False)
df_titles['genre'] = df_titles['genres'].str.split(',').str[0].str.strip()
```

- Memisahkan Role "Actor" dan "Director" supaya mempermudah dalam pemprosesan
  
```
# Pisahkan DIRECTOR
directors = df_credits[df_credits['role'] == 'DIRECTOR'].groupby('id')['name'].first().reset_index()
directors.columns = ['id', 'director']
print(directors.head())

actors = df_credits[df_credits['role'] == 'ACTOR'].groupby('id')['name'].apply(lambda x: ', '.join(x)).reset_index()
actors.columns = ['id', 'cast']
print(actors.head())

df = df.merge(directors, on='id', how='left')
```

- Duplikat Dataset dan Merge director dan actor
Menduplikat dataset bertujuan untuk backup apabila terdapat kekacauan data akibat perubahan.
Kemudian menggabungkan kolom direktor dan kolom actor ke df_clean.

```
df_clean=df_titles.copy()

df_clean = df_clean.merge(directors, on='id', how='left')
df_clean = df_clean.merge(actors, on='id', how='left')
```

### Feature Engineering
Menggabungkan berbagai kolom teks (seperti title, director, cast, genres, dan description) menjadi satu kolom baru content. Hal ini dilakukan untuk memberikan representasi teks yang lebih komprehensif tentang film atau acara yang ada. Dan membuat fitur gabungan untuk pendekatan Content-Based Filtering

```
# Duplikat data
df_CBF=df_clean.copy()

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

### Model Development Collaborative Filtering
#### Imitasi UserId dan Rating
Dari penelusuran dataset sumber terbuka ini bahwa belum memiliki User_Id yang di dapatkan namun Untuk Rating hanya mendapatkan data rata-rata pada imdb_score. Kemudian dari dataset lain yang ada ratingnya memiliki size file yang besar.

- Membuat User_Id secara acak sebanyak 1000 pengguna untuk setiap film (id) dari 500 film yang tersedia di kolom id.

- Membuat kolom Rating yang mengikuti nilai imdb_score dari masing-masing film (menambah variasi kecil agar tidak seragam persis).

Dari tahapan itu, dapat menciptakan data ideal untuk Collaborative Filtering.

#### Mengubah kolom 'user_id' dan 'id' dengan format string

Agar bisa digunakan dalam Collaborative Filtering (CF), terutama dengan algoritma berbasis matrix factorization
```
# mengubah kolom "user_id" menjadi tipe data category
df_CF['user_id']=df_CF['user_id'].astype('category').cat.codes

movie_id_mapping = dict(enumerate(df_CF['id'].astype('category').cat.categories))
df_CF['id'] = df_CF['id'].astype('category').cat.codes
```

#### Split data menjadi data latih dan uji
Memisahkan data pelatihan dan pengujian, merupakan langkah dalam proses training dan pengujian model.

Fungsi train_test_split dari pustaka scikit-learn digunakan untuk membagi dataset menjadi dua bagian, yaitu:

- Data pelatihan (training set) → digunakan untuk melatih model
- Data pengujian (test set) → digunakan untuk menguji performa model

```
# Bagi data
train_data, test_data = train_test_split(df_CF, test_size=0.2, random_state=42)
```

- Membagi dataset df_CF menjadi:
    - 80% untuk pelatihan (train_data)
    - 20% untuk pengujian (test_data)
  
- Parameter random_state=42 memastikan hasilnya reproducible
  
#### Persiapan input untuk model
menyiapkan data pelatihan dan data pengujian dalam bentuk array numerik untuk dimasukkan ke dalam model
  
    ```
    # Ambil input dan konversi ke int32
    train_user_input = train_data['user_id'].values.astype(np.int32)
    train_item_input = train_data['id'].values.astype(np.int32)
    train_ratings = train_data['rating'].values.astype(np.float32)
    
    test_user_input = test_data['user_id'].values.astype(np.int32)
    test_item_input = test_data['id'].values.astype(np.int32)
    test_ratings = test_data['rating'].values.astype(np.float32)
    ```


## Modeling
### 1. Model Development Content Based Filtering

Digunakan untuk merekomendasikan film atau TV show berdasarkan kemiripan konten seperti genre, deskripsi, sutradara, dll. Dengan menggunakan pendekatan matrix factorization melalui embedding layer dan neural network sederhana.

- Parameter :
  
a. Vectorizer: `TfidfVectorizer(stop_words='english')`
b. Similarity Measure: `cosine_similarity(tfidf_matrix)`
    
- Kelebihan :
  
a. Personal dan tidak memerlukan data pengguna lain.
b. Bekerja baik untuk pengguna baru.
  
- Kekurangan :
  
a. Rentan terhadap rekomendasi monoton (konten terlalu mirip).
b. Bergantung pada kualitas metadata.

A) Menghitung kemiripan antar film berdasarkan kontennya, menggunakan Cosine Similarity.

```
# Hitung kemiripan antar konten
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat mapping antara judul dan index DataFrame
indices = pd.Series(df_clean.index, index=df_CBF['title'].str.lower()).drop_duplicates()
```

tfidf_matrix adalah matriks fitur dari teks (misalnya sinopsis film) yang telah diubah menggunakan TF-IDF Vectorizer.

B. Matriks Kemiripan Antar Judul

Digunakan untuk melihat atau mengakses kemiripan antar film berdasarkan judul secara langsung.

| **title**                               | **Open Graves** | **Digging to Death** | **The Grand Tour** | **Maze** | **Sita Ramam** |
| --------------------------------------- | --------------- | -------------------- | ------------------ | -------- | -------------- |
| **Open Graves**                         | 1.000000        | 0.005245             | 0.007073           | 0.007780 | 0.000000       |
| **Digging to Death**                    | 0.005245        | 1.000000             | 0.018771           | 0.003384 | 0.006124       |
| **The Grand Tour**                      | 0.007073        | 0.018771             | 1.000000           | 0.022093 | 0.000000       |
| **Maze**                                | 0.007780        | 0.003384             | 0.022093           | 1.000000 | 0.000933       |
| **Sita Ramam**                          | 0.000000        | 0.006124             | 0.000000           | 0.000933 | 1.000000       |
| **All Through the House**               | 0.027390        | 0.022807             | 0.000000           | 0.000000 | 0.000000       |
| **Avenge the Crows**                    | 0.001659        | 0.002299             | 0.000000           | 0.008271 | 0.003750       |
| **Befikre**                             | 0.000000        | 0.010009             | 0.003108           | 0.001402 | 0.011051       |
| **The Adventures of Ozzie and Harriet** | 0.000000        | 0.011166             | 0.002183           | 0.000985 | 0.000477       |
| **Putham Pudhu Kaalai**                 | 0.000000        | 0.000000             | 0.000000           | 0.000970 | 0.061185       |

#### Sistem Rekomendasi
Membuat sistem rekomendasi berbasis konten (Content-Based Filtering), yang merekomendasikan film atau acara TV yang mirip dengan sebuah judul tertentu, berdasarkan kemiripan teks (misalnya sinopsis) yang dihitung dengan cosine similarity.

| **No** | **Title**             | **Type** | **Genres**                      | **Description**                                            | **Similarity Score** |
| -----: | --------------------- | -------- | ------------------------------- | ---------------------------------------------------------- | -------------------: |
|      1 | Bullitt County        | MOVIE    | action, drama, thriller         | An action/thriller set in 1977 about four friends...       |             0.115214 |
|      2 | At Granny's House     | MOVIE    | thriller                        | A typical Midwest house. A sweet little old lady...        |             0.112151 |
|      3 | Bleed                 | MOVIE    | horror, thriller                | A naïve young girl desperate to fit in with her...         |             0.108470 |
|      4 | Karma                 | MOVIE    | drama, thriller, comedy, action | When senior police inspector Vishwa Pratap Singh...        |             0.098885 |
|      5 | The Handler           | MOVIE    | action, drama, thriller, crime  | After throwing a job, an ex-Marine seeks refuge...         |             0.097815 |
|      6 | Sniper Corpse         | MOVIE    | horror                          | The undead are former militia soldiers that are...         |             0.095206 |
|      7 | The Bat               | MOVIE    | horror, thriller                | Mystery writer Cornelia Van Gorder has rented...           |             0.094364 |
|      8 | House on Haunted Hill | MOVIE    | horror, crime                   | Frederick Loren has invited five strangers to...           |             0.089328 |
|      9 | Devil in the Flesh    | MOVIE    | horror, thriller                | When her mother is killed in a mysterious house...         |             0.084746 |
|     10 | Three Pines           | SHOW     | crime, drama                    | Chief Inspector Armand Gamache and his team investigate... |             0.084620 |

#### Evaluasi Model Development Content Based Filtering
Untuk mengevaluasi seberapa baik sistem rekomendasi, khususnya sistem rekomendasi berbasis konten (Content-Based Filtering / CBF) menggunakan metrik informasi retrieval:

- Precision@K
- Recall@K
- F1-Score@K

Berikut hasil dari evaluate_recommendation_system

Evaluation for: 'Bleed'

Top-10 Recommendations: ['DIVE!!', 'Digging to Death', 'Girl, Chill', 'Twinsanity', 'Cave Club', 'Chicago Massacre: Richard Speck', 'Seven Alone', 'Hellblock 13', 'Devil in the Flesh', 'Erik Terrell: Live at the Helium Comedy Club']

Ground Truth: ['Putham Pudhu Kaalai ', 'Digging to Death', 'All Through the House', 'Sita Ramam']

Precision@10: 0.1

Recall@10:    0.25

F1-Score@10:  0.1429

{'precision': 0.1, 'recall': 0.25, 'f1_score': 0.1429}

Bedasarkan hasil evaluasi ini sistem rekomendasi masih perlu ditingkatkan, terutama dalam memahami konteks film yang relevan secara semantik.

### 2. Model Development Collaborative Filtering
Digunakan untuk merekomendasikan film atau TV show berdasarkan kemiripan konten seperti genre, deskripsi, sutradara, dll.

- Parameter :
-- Embedding(input_dim=num_users, output_dim=50)
-- Flatten()
-- Concatenate()
-- Dense(128, activation='relu')(merged)
-- Dropout(0.3)(x)
-- Dense(64, activation='relu')(x)
-- Dense(1)(x)

- Parameter Pelatihan:
-- Optimizer: Adam
-- Loss: Mean Squared Error
-- Epochs: 20
-- Batch Size: 64

- Kelebihan :
-- Mampu menangkap preferensi kompleks antar pengguna.
-- Memberikan rekomendasi yang beragam.
      
- Kekurangan :
-- Tidak bekerja optimal untuk pengguna/item baru (cold-start).
-- Membutuhkan volume data besar dan preprocessing tambahan.

#### Neural Collaborative Filtering (NCF)    
Membangun model prediksi rating berdasarkan user ID dan item (movie) ID dengan pendekatan embedding dan dense layers. Terdapat fungsi masing masing dalam NFC :

| Fungsi                       | Penjelasan                                                      |
| ---------------------------- | --------------------------------------------------------------- |
| Membuat embedding            | Mewakili user dan item sebagai vektor yang bisa dipelajari      |
| Bangun jaringan neural       | Gunakan dense layers untuk prediksi rating                      |
| Compile model                | Melatih dengan optimisasi dan metrik evaluasi        |
| RMSE sebagai metrik tambahan | Memberi gambaran lebih jelas terhadap kesalahan prediksi rating |


**Model: "functional"** 

| **Layer (Type)**             | **Output Shape** | **Param #** | **Connected to**                    |
| ---------------------------- | ---------------- | ----------- | ----------------------------------- |
| input\_layer (InputLayer)    | (None, 1)        | 0           | -                                   |
| input\_layer\_1 (InputLayer) | (None, 1)        | 0           | -                                   |
| embedding (Embedding)        | (None, 1, 50)    | 50,000      | input\_layer\[0]\[0]                |
| embedding\_1 (Embedding)     | (None, 1, 50)    | 25,000      | input\_layer\_1\[0]\[0]             |
| flatten (Flatten)            | (None, 50)       | 0           | embedding\[0]\[0]                   |
| flatten\_1 (Flatten)         | (None, 50)       | 0           | embedding\_1\[0]\[0]                |
| concatenate (Concatenate)    | (None, 100)      | 0           | flatten\[0]\[0], flatten\_1\[0]\[0] |
| dense (Dense)                | (None, 128)      | 12,928      | concatenate\[0]\[0]                 |
| dropout (Dropout)            | (None, 128)      | 0           | dense\[0]\[0]                       |
| dense\_1 (Dense)             | (None, 64)       | 8,256       | dropout\[0]\[0]                     |
| dense\_2 (Dense)             | (None, 1)        | 65          | dense\_1\[0]\[0]                    |


- Training model Collaborative Filtering
Digunakan untuk melatih model rekomendasi berbasis Neural Collaborative Filtering (NCF) yang sudah dibuat sebelumnya.

![Epoch](https://github.com/user-attachments/assets/8ddbfa69-2d90-43b3-9f51-b664b286cbf2)


Dari hasil yang didapatkan bahwa model telah berhasil dilatih secara stabil, dan saat ini mencapai rata-rata kesalahan prediksi rating sebesar ±1.31. Ini menunjukkan model sudah belajar cukup baik, namun masih memiliki ruang untuk ditingkatkan

- Fungsi rekomendasi film berdasarkan prediksi rating
Bertujuan untuk menghasilkan rekomendasi film untuk pengguna tertentu berdasarkan prediksi rating dari model Neural Collaborative Filtering (NCF) yang telah dilatih sebelumnya.

Top 10 rekomendasi film untuk user 445:
| **No.** | **Title**                   | **Genres**                                    | **Type** | **IMDb Score** | **TMDb Score** |
| ------: | --------------------------- | --------------------------------------------- | -------- | -------------- | -------------- |
|       1 | It's a Wonderful Life       | drama, family, fantasy, romance, comedy       | MOVIE    | 8.6            | 8.261          |
|       2 | The Three Stooges           | comedy, family                                | SHOW     | 8.5            | 7.6            |
|       3 | The Jack Benny Program      | comedy                                        | SHOW     | 8.6            | 7.5            |
|       4 | The Best Years of Our Lives | drama, romance, war                           | MOVIE    | 8.1            | 7.838          |
|       5 | The Little Foxes            | drama, romance                                | MOVIE    | 7.9            | 7.549          |
|       6 | The Gold Rush               | drama, comedy, romance, western, family       | MOVIE    | 8.1            | 8.03           |
|       7 | The General                 | comedy, drama, action, war, western, european | MOVIE    | 8.1            | 8.009          |
|       8 | My Man Godfrey              | comedy, drama, romance                        | MOVIE    | 8.0            | 7.56           |
|       9 | Scarlet Street              | drama, thriller, crime                        | MOVIE    | 7.8            | 7.6            |
|      10 | What's My Line?             | reality, family                               | SHOW     | 8.5            | 7.2            |


#### Evaluasi Collaborative Filtering
![EvaluationCF](https://github.com/user-attachments/assets/9714d906-172f-439f-a199-39ca439fcb3c)


### Improvement Process
- Collaborative Filtering mampu menyediakan rekomendasi yang lebih variatif dan personal dengan memanfaatkan pola interaksi pengguna.
- Content-Based Filtering memberikan rekomendasi yang konsisten bagi pengguna dengan riwayat preferensi yang jelas, namun cenderung menghasilkan rekomendasi yang kurang beragam karena fokus pada konten serupa


### Model Terbaik pada Solution Statement
Bedasarkan hasil modeling bahwa Collaborative Filtering kurang baik untuk digunakan sebagai model utama karena belum mampu memberikan prediksi yang lebih akurat terhadap preferensi pengguna. Sementara itu, Content-Based Filtering dapat digunakan sebagai model utama karena hasil loss yang rendah.

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

Top 10 rekomendasi film untuk user 445:
| **No.** | **Title**                   | **Genres**                                    | **Type** | **IMDb Score** | **TMDb Score** |
| ------: | --------------------------- | --------------------------------------------- | -------- | -------------- | -------------- |
|       1 | It's a Wonderful Life       | drama, family, fantasy, romance, comedy       | MOVIE    | 8.6            | 8.261          |
|       2 | The Three Stooges           | comedy, family                                | SHOW     | 8.5            | 7.6            |
|       3 | The Jack Benny Program      | comedy                                        | SHOW     | 8.6            | 7.5            |
|       4 | The Best Years of Our Lives | drama, romance, war                           | MOVIE    | 8.1            | 7.838          |
|       5 | The Little Foxes            | drama, romance                                | MOVIE    | 7.9            | 7.549          |
|       6 | The Gold Rush               | drama, comedy, romance, western, family       | MOVIE    | 8.1            | 8.03           |
|       7 | The General                 | comedy, drama, action, war, western, european | MOVIE    | 8.1            | 8.009          |
|       8 | My Man Godfrey              | comedy, drama, romance                        | MOVIE    | 8.0            | 7.56           |
|       9 | Scarlet Street              | drama, thriller, crime                        | MOVIE    | 7.8            | 7.6            |
|      10 | What's My Line?             | reality, family                               | SHOW     | 8.5            | 7.2            |



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
