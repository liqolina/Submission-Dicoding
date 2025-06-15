# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding  
Jaya Jaya Institut merupakan lembaga pendidikan tinggi berbasis edutech. Namun Jaya Jaya Institut memiliki permasalahan dalam mempertahankan mahasiswa sampai lulus. Dengan tingginya mahasiswa dropout dapat mempengaruhi kinerja akademik Jaya Jaya Institut. Namun saat ini Jaya Jaya Institut belum memiliki solusi untuk memantau resiko dropout dan mengurangi jumlah mahasiswa dropout. Oleh karena itu, Jaya Jaya Institut membutuhkan sistem yang mampu dalam memprediksi kemungkinan mahasiswa dropout.

### Permasalahan Bisnis  
Beberapa faktor yang menjadi akar permasalahan antara lain:
1. Kurangnya Sistem Prediktif

   Hingga saat ini, institusi belum memiliki sistem yang mampu mengidentifikasi secara dini mahasiswa dengan risiko tinggi untuk berhenti kuliah. Akibatnya, intervensi sering dilakukan secara terlambat dan kurang personal.

2. Minimnya Visibilitas Risiko

   Tidak tersedia dashboard real-time yang memungkinkan pemantauan risiko berdasarkan berbagai segmen penting seperti program studi, status pernikahan, jenis beasiswa, dan kondisi sosial lainnya.

3. Pengambilan Keputusan Tidak Berbasis Data

   Selama ini, proses evaluasi kinerja mahasiswa dan pengambilan keputusan strategis belum sepenuhnya didasarkan pada data historis yang akurat dan terstruktur. Kurangnya alat bantu analitik dan visualisasi data juga menyulitkan dalam memahami pola-pola penting yang tersembunyi.

### Cakupan Proyek
Pada proyek ini, dilakukan beberapa tahap utama untuk menyelesaikan permasalahan dropout mahasiswa di Jaya Jaya Institut, yaitu:

1. Business Understanding

Pada tahap ini dilakukan identifikasi menyeluruh terhadap konteks permasalahan dropout yang berdampak pada mutu pendidikan, reputasi institusi, dan efisiensi operasional. Pengembangan sistem prediktif yang dapat mendukung pelaksanaan intervensi secara dini dan tepat sasaran oleh pihak manajemen.

2. Data Understanding dan Preparation

Dilakukan eksplorasi terhadap data historis mahasiswa untuk memperoleh pemahaman awal mengenai struktur dan karakteristik data. Proses ini mencakup identifikasi dan penanganan nilai hilang (missing values), pencilan (outliers), serta penerapan teknik transformasi fitur seperti encoding dan feature engineering, guna memastikan data dalam kondisi layak untuk digunakan dalam tahap pemodelan.

3. Exploratory Data Analysis (EDA)

Tahap ini bertujuan untuk menggali pola dan tren dropout mahasiswa berdasarkan variabel-variabel utama seperti tingkat pendidikan orang tua, status beasiswa, jenis kelamin, capaian akademik, serta latar belakang sosial ekonomi. Hasil analisis ini digunakan untuk menghasilkan wawasan awal yang relevan dan mendukung proses pemodelan prediktif.

4. Modeling

Pengembangan model prediktif dilakukan dengan membandingkan beberapa algoritma machine learning, antara lain Logistic Regression, Random Forest, Gradient Boosting, dan SVM. Kinerja masing-masing model dievaluasi menggunakan metrik kuantitatif seperti akurasi, F1-score, precision, dan recall guna menentukan model yang paling optimal.

5. Evaluation

Model dengan performa terbaik dipilih berdasarkan hasil evaluasi yang objektif. Selanjutnya dilakukan interpretasi terhadap variabel-variabel yang paling berpengaruh terhadap probabilitas mahasiswa mengalami dropout, sebagai dasar bagi perumusan strategi intervensi.

6. Script Prediction & Deployment Preparation

Disusun skrip inferensi berbasis Python yang mengintegrasikan model terpilih dalam format .pkl, serta proses preprocessing yang konsisten dengan menggunakan scaler.pkl dan feature_columns.pkl. Model ini kemudian diintegrasikan ke dalam sebuah aplikasi web berbasis Streamlit yang memungkinkan pengguna memasukkan data mahasiswa secara manual untuk memperoleh hasil prediksi. Aplikasi telah berhasil di-deploy melalui Streamlit Community Cloud dan dapat diakses secara daring oleh pihak pemangku kepentingan.

7. Dashboard Development

Dibuat dashboard interaktif menggunakan platform Looker Studio yang terhubung dengan basis data Supabase. Dashboard ini menyajikan visualisasi statistik dan tren dropout mahasiswa secara real-time berdasarkan dimensi seperti jenis kelamin, status beasiswa, nilai akademik, dan parameter relevan lainnya, guna mendukung proses monitoring dan pengambilan keputusan.

8. Recommendation

Berdasarkan hasil analisis dan pemodelan, dirumuskan rekomendasi kebijakan berbasis data kepada pihak institusi, di antaranya: penguatan program beasiswa bagi kelompok rentan, optimalisasi sistem pemantauan akademik, serta penyediaan intervensi sosial dan dukungan psikologis bagi mahasiswa dengan tingkat risiko tinggi.

## Persiapan  


**Sumber data**:  
Dataset yang digunakan dalam proyek ini berasal dari data internal mahasiswa Jaya Jaya Institut, yang mencakup informasi demografis, rekam jejak akademik, serta kondisi sosial dan ekonomi masing-masing mahasiswa.

Dataset dapat diakses melalui tautan berikut:

[Link Github](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)


**Setup environment**:  

Untuk menjalankan aplikasi dan model prediksi risiko dropout secara optimal, diperlukan proses penyiapan lingkungan kerja (environment) dengan konfigurasi yang sesuai. Adapun tahapan yang perlu dilakukan adalah sebagai berikut:

1. Membuat Environment Baru dengan Conda

   Langkah pertama adalah membuat lingkungan baru bernama dropout-prediction menggunakan Python versi 3.9.15. Jalankan perintah berikut melalui terminal atau command prompt:

   ```
   conda create -n dropout-prediction python=3.11
   ```

2. Mengaktifkan Environment

   Setelah environment berhasil dibuat, aktifkan dengan perintah berikut:
   ```
   conda activate dropout-prediction
   ```
   
3. Menginstal Dependensi

   Setelah environment aktif, instal semua dependensi yang dibutuhkan untuk menjalankan aplikasi dan model prediktif. Pastikan file requirements.txt telah tersedia dalam direktori proyek, kemudian jalankan perintah berikut:
   
   ```
   pip install -r requirements.txt
   ```

## Business Dashboard

Dashboard interaktif dikembangkan menggunakan Looker Studio dengan tujuan membantu manajemen dalam memantau tren dropout mahasiswa secara real-time. Dashboard ini menyediakan analisis mendalam berdasarkan berbagai faktor penting, antara lain usia mahasiswa, latar belakang pendidikan orang tua, status pembayaran biaya kuliah, serta variabel relevan lainnya yang berkontribusi terhadap risiko dropout.

Melalui dashboard ini, manajemen dapat memperoleh insight yang lebih komprehensif untuk mendukung pengambilan keputusan strategis dan perencanaan intervensi yang lebih tepat sasaran.

Dashboard dapat diakses melalui tautan berikut:
[Link Dashboard Looker Studio](https://lookerstudio.google.com/reporting/a31a819a-8291-4f65-ae42-4655f3b2ff76)

Dashboard-1
![lutfi_hermawan-dashboard-1](https://github.com/user-attachments/assets/a42e02f7-faed-4326-a7d3-e49b064a4327)

Dashboard-2 
![lutfi_hermawan-dashboard-2](https://github.com/user-attachments/assets/cb78c835-8dec-449b-a787-d215ecf82de3)


## Menjalankan Sistem Machine Learning
### Prototipe Sistem Prediksi Dropout
Prototipe sistem prediksi dropout dikembangkan menggunakan platform Streamlit, yang memungkinkan pengguna untuk memasukkan data mahasiswa secara langsung dan memperoleh prediksi risiko dropout secara real-time melalui antarmuka web yang interaktif.

### Akses Prototipe
Pengguna dapat mencoba prototipe ini secara daring melalui tautan berikut:
[Link Dashboard Streamlit](https://submission-dicoding-vx8dvmycvtnq3zstrhpisz.streamlit.app/)

Instruksi Menjalankan Prototipe Secara Lokal
Untuk menjalankan aplikasi di lingkungan lokal, jalankan perintah berikut pada terminal setelah mengatur lingkungan yang diperlukan:

```
streamlit run app.py
```

## Conclusion
Model Random Forest menunjukkan performa terbaik dengan akurasi sekitar 78% serta recall tertinggi dalam mendeteksi mahasiswa yang berisiko dropout. Model ini secara konsisten unggul dalam mengidentifikasi kelas dropout dibandingkan dengan model lain yang diuji. Selain itu, dashboard yang dibuat juga sangat membantu dalam memberikan informasi terkini terkait status mahasiswa, sehingga memudahkan pengambilan keputusan yang tepat dan cepat untuk mengatasi potensi dropout. Dengan demikian, kombinasi model Random Forest dan dashboard interaktif dapat menjadi alat yang efektif dalam mendukung upaya peningkatan retensi mahasiswa.

## Rekomendasi Action Items
Berdasarkan hasil analisis dan insight yang diperoleh dari dashboard Student Dropout Monitoring di Jaya Jaya Institut, berikut beberapa strategi yang disarankan untuk mengurangi tingkat dropout dan meningkatkan tingkat kelulusan mahasiswa:

1. Pemantauan Dini dan Intervensi Proaktif
Memberikan bimbingan belajar, mentoring, dan program remedial untuk mahasiswa dengan nilai masuk dan rata-rata semester rendah agar dapat memperbaiki prestasi sejak dini.

2. Pendanaan Pendidikan yang Fleksibel
Implementasi program pendanaan pendidikan cicilan 0% dan bantuan keuangan darurat untuk mahasiswa mengalami krisis ekonomi mendadak.

3. Keterlibatan Orang Tua dan Komunitas
Mengadakan workshop dan seminar khusus untuk orang tua guna meningkatkan pemahaman mereka tentang pendidikan dan cara mendukung anak secara efektif, sehingga mahasiswa mendapat dukungan sosial dan emosional yang kuat. 

4. Pelayanan konseling bagi mahasiswa
Menyediakan layanan konseling sosial dan psikologis bagi mahasiswa yang menunjukkan tanda-tanda tekanan sosial atau emosional.

5. Program Beasiswa dan Bantuan Finansial
Menawarkan beasiswa mini/subsidi uang kuliah untuk mahasiswa berisiko
