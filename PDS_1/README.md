# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Perusahaan Jaya Jaya Maju

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Perusahaan ini memiliki lebih dari 1.000 karyawan yang tersebar di berbagai wilayah di seluruh Indonesia. Meskipun telah berkembang menjadi perusahaan berskala besar, Jaya Jaya Maju masih menghadapi tantangan dalam pengelolaan sumber daya manusia. Salah satu dampak dari tantangan tersebut adalah tingginya tingkat attrition yang saat ini mencapai lebih dari 10%. 

Tingginya attrition berdampak negatif terhadap berbagai aspek operasional perusahaan seperti biaya rekrutmen dan pelatihan, mempengaruhi produktivitas dan semangat kerja tim secara keseluruhan dan kontinuitas pekerjaan. 

Untuk mengatasi permasalahan ini, diperlukan upaya strategis berupa identifikasi berbagai faktor yang memengaruhi tingginya tingkat attrition, guna mendukung pengambilan keputusan yang lebih tepat dalam pengelolaan dan retensi karyawan.

### Permasalahan Bisnis

Perusahaan Multinasional Jaya Jaya Maju menghadapi tantangan terkait tingginya tingkat attrition dalam beberapa tahun terakhir. Tingginya attrition dapat berdampak negatif terhadap berbagai aspek operasional perusahaan, antara lain meningkatnya biaya rekrutmen dan pelatihan, terganggunya kontinuitas pekerjaan, serta menurunnya produktivitas dan semangat kerja tim secara keseluruhan. 

Selain itu, hingga saat ini perusahaan Jaya Jaya Maju belum memiliki sistem yang mampu mengidentifikasi penyebab atau faktor-faktor utama yang berkontribusi terhadap tingginya tingkat attrition karyawan. Ketiadaan sistem ini menyebabkan perusahaan kesulitan dalam memahami pola atau tren yang menyebabkan karyawan memilih untuk keluar.

Lebih lanjut, perusahaan juga belum memiliki mekanisme pemantauan (monitoring) yang terintegrasi untuk mengawasi dinamika faktor-faktor internal yang berpotensi memengaruhi keputusan karyawan untuk meninggalkan perusahaan. Hal ini semakin memperumit upaya manajemen dalam merancang strategi retensi karyawan yang efektif dan berbasis data.

### Cakupan Proyek

Pada proyek ini, dilakukan beberapa tahap utama, yaitu:

**1. Business Understanding**

Tahap ini bertujuan untuk mengidentifikasi konteks bisnis secara menyeluruh dan mendefinisikan permasalahan yang dihadapi perusahaan yaitu attrition karyawan.

**2. Data Understanding & Preparation**

Melakukan eksplorasi awal terhadap dataset karyawan, termasuk penanganan missing values dan outliers, serta melakukan transformasi fitur yang diperlukan guna mendukung analisis dan pemodelan data secara optimal.

**3. Exploratory Data Analysis (EDA)**

Visualisasi dilakukan terhadap distribusi berbagai variabel, seperti pendapatan, masa kerja, jarak ke kantor, dan variabel lainnya. Melalui visualisasi ini, dapat dianalisis pola dan tren yang berkaitan dengan attrition berdasarkan data masing-masing variabel, sehingga memungkinkan identifikasi faktor-faktor yang berpotensi memengaruhi tingkat attrition karyawan.

**4. Modeling**

Mengembangkan beberapa model prediktif untuk memproyeksikan kemungkinan attrition, antara lain menggunakan algoritma Logistic Regression, Random Forest, XGBoost, Gradient Boosting, dan SVM. Selanjutnya, dilakukan pengaturan parameter pelatihan model untuk memastikan bahwa performa yang dihasilkan akurat dalam memprediksi tingkat attrition. Performa masing-masing model dibandingkan menggunakan metrik evaluasi seperti akurasi, F1-score, dan AUC.

**5. Evaluation**

Pemilihan model terbaik dilakukan berdasarkan hasil evaluasi terhadap sejumlah metrik kinerja. Selain itu, dilakukan interpretasi terhadap fitur-fitur yang paling berpengaruh dalam proses prediksi attrition, guna memberikan wawasan yang lebih mendalam mengenai faktor-faktor utama yang mendorong terjadinya attrition di perusahaan.

**6. Script Prediction & Deployment Preparation**

Penyusunan script Python dilakukan untuk mengotomatiskan proses prediksi attrition. Model terbaik disimpan dalam format file .pkl agar dapat dengan mudah diintegrasikan ke dalam sistem.

**7. Dashboard Development**

Dashboard interaktif dikembangkan menggunakan Looker Studio untuk memvisualisasikan insight dan metrik utama. Dashboard ini bertujuan mendukung pengambilan keputusan berbasis data dengan informasi yang mudah dipahami dan dapat diakses oleh manajer departemen HR.

**8. Recommendation**

Rekomendasi strategis berbasis data disusun dan disampaikan kepada tim departemen Human Resources (HR) sebagai upaya untuk meningkatkan tingkat retensi karyawan serta menurunkan angka attrition. Rekomendasi ini didasarkan pada hasil analisis mendalam terhadap faktor-faktor yang memengaruhi attrition, sehingga dapat mendukung perumusan kebijakan yang lebih tepat sasaran.

### Persiapan

#### A. Sumber data:
   
- Sumber data dataset dapat diakses melalui tautan berikut:
  
   [Github](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)
   
#### B. Setup environment:
   
   **1. Conda Environment**
   - **Buat dan Aktifkan Environment Conda**
     
        Membuat environment Conda baru dengan Python versi 3.11.13
     
         conda create -n notebook-env python=3.11.13 -y
         
        Setelah environment berhasil dibuat, aktifkan dengan perintah berikut:

         conda activate notebook-env
     
   - **Install Requirements**

         pip install -r requirements.txt
   
   - **Jalankan Notebook**
     
         jupyter notebook

   **2. Predict Attrition with Modelling**

   - Untuk menjalankan proses prediksi, gunakan perintah berikut di terminal:
    
         python predict.py
     
## Business Dashboard

Business dashboard yang telah dibuat dalam proyek ini merupakan sebuah dashboard yang dikembangkan menggunakan Looker Studio. Tujuan utamanya adalah untuk menyajikan informasi dan insight penting terkait attrition karyawan secara visual dan mudah dipahami, guna mendukung pengambilan keputusan berbasis data oleh tim manajemen, khususnya departemen HR.

### Fitur Utama Dashboard:

1. Tingkat Attrition Keseluruhan
2. Distribusi Attrition Berdasarkan Variabel Penting
3. Fitur-Fitur yang Mempengaruhi Attrition
4. Filter Interaktif

### Manfaat:
1. Meningkatkan visibilitas terhadap penyebab utama attrition.
2. Membantu tim HR dalam memonitor efektivitas strategi retensi.
3. Memungkinkan tindakan proaktif terhadap area-area berisiko tinggi.

### Visualize Dashboard
Dashboard dapat diakses pada link berikut :

[Looker Studio](https://lookerstudio.google.com/reporting/7497e034-bc96-4ff8-87bd-a2282a73d1bf)

## Conclusion

Proyek ini bertujuan untuk membantu perusahaan Jaya Jaya Maju dalam mengidentifikasi dan memahami penyebab utama tingginya tingkat attrition karyawan, serta mengembangkan sistem prediksi berbasis data yang lebih efektif. 

Melalui tahapan Business Understanding, EDA, pemodelan machine learning, dan dashboard visualisasi, ditemukan bahwa Usia, jumlah total tahun pengalaman kerja, tingkat jabatan, opsi saham, dan pendapatan bulanan mempengaruhi karyawan untuk bertahan di perusahaan. 

Dengan pemodelan dan Business Dashboard dapat membantu departemen Human Resources (HR) dalam memonitor faktor-faktor yang mempengaruhi keluarnya karyawan dan mengambil tindak pencegahan dengan tepat.

### Rekomendasi Action Items (Optional)

Bedasarkan hasil analisis, berikut rekomendasi aksi untuk departemen Human Resources (HR) Jaya Jaya Maju :

- Perkuat Hubungan Karyawan dengan Atasan Langsung (LoyaltyToManager).
     Dengan hubungan baik antara atasan dengan karyawan akan menciptakan lingkungan kerja yang positif. Salah satu cara untuk membangun hubungan ini adalah dengan rutin melakukan survei umpan balik, sehingga karyawan merasa didengar dan dihargai."
- Tingkatkan Program Pelatihan yang Tepat Sasaran.
     Dengan memberikan pelatihan yang relevan dan sesuai kebutuhan karyawan sehingga pengembangan yang didapat membantu dalam pekerjaan  dan jalur kariernya, sehingga  meningkatkan loyalitas dan mengurangi kemungkinan karyawan untuk keluar dari perusahaan.
- Fokus pada Retensi Karyawan Muda dan Baru Bergabung.
     Dengan menyelenggarakan program onboarding dan pengembangan khusus yang dirancang untuk karyawan usia muda serta mereka yang baru bergabung, agar mereka merasa lebih cepat beradaptasi, mendapatkan dukungan yang memadai, dan memiliki peluang pengembangan karier yang jelas
