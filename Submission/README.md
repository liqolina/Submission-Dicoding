# Flower Classification Menggunakan Convolutional Neural Networks (CNN)

## Latar Belakang Proyek

Proyek ini bertujuan untuk merancang sebuah model Convolutional Neural Network (CNN) yang mampu mengklasifikasikan gambar bunga ke dalam 14 kelas berbeda berdasarkan dataset yang tersedia. Dataset tersebut dibagi menjadi tiga bagian, yaitu data pelatihan, validasi, dan pengujian. Model ini juga menggunakan pendekatan transfer learning dengan memanfaatkan arsitektur MobileNetV3Large. Untuk mencegah overfitting, digunakan callback seperti EarlyStopping dan ModelCheckpoint. Model yang telah dilatih diekspor ke dalam format SavedModel, TFLite, dan TFJS agar dapat digunakan pada berbagai platform.

## Dataset
Download Dataset melalui Google Drive

[Google Drive](https://drive.google.com/file/d/1UhgDkbQY_8PGlE98EuiXcP5kWjaiQnOW/view?usp=drive_link)

### Informasi Dataset:
- Dataset Flower Classification terdiri dari 14 kelas bunga yang berbeda.

- Gambar dalam dataset memiliki resolusi yang bervariasi.

### Dataset Splitting:

Dataset dibagi menjadi beberapa subset sebagai berikut:

Training: 70%
Validation: 15%
Testing: 15%

### Data Augmentasi Train :

Data augmentation digunakan untuk meningkatkan keberagaman data pelatihan dengan tujuan meningkatkan kemampuan generalisasi model dan mengurangi overfitting.

### Class Weights :

Class weights digunakan untuk menyeimbangkan distribusi jumlah gambar antar kelas agar model tidak bias terhadap kelas mayoritas.

## Model Architecture

### Tipe Model:

- Menggunakan model bertipe Sequential.

### Layers:

- **Convolutional Layers (Conv2D)**: Lapisan konvolusi bertugas mengekstraksi fitur dari gambar input dengan menerapkan filter (kernel) yang menghasilkan feature maps. Setiap filter mendeteksi pola spesifik seperti tepi, tekstur, atau bentuk tertentu.

- **Pooling Layers**: Lapisan pooling, seperti MaxPooling, digunakan untuk mengurangi dimensi spasial dari feature maps. Ini membantu dalam mengurangi jumlah parameter dan komputasi, serta mengontrol overfitting dengan mempertahankan informasi penting. 
Wikipedia

- **Dropout Layer**: Lapisan dropout secara acak menonaktifkan sejumlah unit selama pelatihan, yang berfungsi untuk mencegah overfitting dengan mengurangi ketergantungan antar neuron.

- **Fully Connected (Dense) Layer**: Setelah proses ekstraksi fitur, lapisan fully connected menggabungkan semua neuron dari lapisan sebelumnya untuk melakukan klasifikasi akhir. 


### Fungsi Aktivasi:

- ReLU (Rectified Linear Unit): Digunakan pada lapisan tersembunyi untuk memperkenalkan non-linearitas, ReLU mengubah nilai negatif menjadi nol, memungkinkan jaringan untuk mempelajari hubungan kompleks dalam data. 
Wikipedia

- Softmax: Diterapkan pada lapisan output untuk mengubah output menjadi distribusi probabilitas atas 14 kelas, memastikan total probabilitas berjumlah 1.

## Training & Evaluation

### Proses Training:

Model dilatih menggunakan metode model.fit() dari Keras, yang merupakan pendekatan standar untuk pelatihan model. Proses pelatihan ini melibatkan beberapa komponen utama:

- Optimizer: Digunakan Adam optimizer, yang merupakan metode optimisasi berbasis stochastic gradient descent dengan estimasi adaptif dari momen orde pertama dan kedua. Adam dikenal karena efisiensinya dalam komputasi dan kinerjanya yang baik pada berbagai masalah pembelajaran mesin. 
TensorFlow

- Fungsi Loss: Menggunakan categorical cross-entropy sebagai fungsi loss, yang sesuai untuk tugas klasifikasi multi-kelas dengan label yang dikodekan secara one-hot.

- Callback:

- EarlyStopping: Callback ini memantau metrik validasi (misalnya, val_loss) dan menghentikan pelatihan jika tidak ada peningkatan setelah sejumlah epoch tertentu, yang ditentukan oleh parameter patience. Hal ini membantu mencegah overfitting dan menghemat sumber daya komputasi. 

- ModelCheckpoint: Callback ini menyimpan model ke file setiap kali terjadi peningkatan pada metrik yang dipantau (misalnya, val_loss). Dengan demikian, model terbaik selama pelatihan dapat disimpan dan digunakan untuk evaluasi atau deployment selanjutnya.

## Visualisasi

### Accuracy and Loss Plots:



## Model Deployment

### Format Model:

Model yang telah dilatih disimpan dalam beberapa format agar bisa digunakan di berbagai platform:

- SavedModel: Format standar TensorFlow yang cocok untuk digunakan di server atau backend aplikasi.

- TensorFlow Lite (TFLite): Format ringan yang dioptimalkan untuk dijalankan di perangkat mobile atau embedded seperti Raspberry Pi.

- TensorFlow.js (TFJS): Format yang memungkinkan model dijalankan langsung di browser menggunakan JavaScript, cocok untuk aplikasi web.
