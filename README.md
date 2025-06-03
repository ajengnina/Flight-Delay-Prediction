# Prediksi Keterlambatan Penerbangan

## Domain Proyek

Keterlambatan penerbangan adalah permasalahan serius yang mempengaruhi efisiensi operasional dan kepuasan penumpang. Menurut Federal Aviation Administration (FAA), keterlambatan penerbangan menyumbang kerugian miliaran dolar setiap tahunnya. Oleh karena itu, penting untuk membangun sistem prediksi yang dapat mengidentifikasi kemungkinan keterlambatan agar maskapai dan penumpang dapat melakukan tindakan preventif.

---

## Business Understanding

### Problem Statement
Permasalahan yang ingin diselesaikan adalah bagaimana mengidentifikasi faktor-faktor yang memengaruhi keterlambatan penerbangan dan membangun model prediktif yang dapat mengklasifikasikan apakah sebuah penerbangan akan mengalami keterlambatan (delay) lebih dari 15 menit atau tidak. Keterlambatan penerbangan berdampak besar terhadap operasional maskapai, kepuasan pelanggan, dan efisiensi sistem transportasi udara. Oleh karena itu, diperlukan sistem yang mampu memberikan prediksi keterlambatan secara akurat untuk mendukung pengambilan keputusan yang lebih baik.

### Goals
Membangun model klasifikasi yang mampu memprediksi apakah penerbangan akan mengalami keterlambatan berdasarkan berbagai fitur historis dan operasional penerbangan.

### Solution Statement

- Membangun dua model: **Logistic Regression** dan **Random Forest Classifier**.
- Melakukan tuning parameter menggunakan GridSearchCV pada Random Forest.
- Menilai performa model menggunakan metrik: accuracy, precision, recall, dan F1-score.

---

## Data Understanding
### Sumber Data
Dataset yang digunakan berasal dari Kaggle - US Flight Delays Dataset. Dataset ini berisi informasi terkait data penerbangan komersial di Amerika Serikat, termasuk waktu keberangkatan, keterlambatan, serta faktor-faktor eksternal yang memengaruhi.

### Sumber dataset:
https://www.kaggle.com/datasets/arvindnagaonkar/flight-delay

### Ukuran Dataset
Jumlah baris: 1.936.758(namun saya hanya mengambil 10.000 data saja)
Jumlah kolom: 31

### Kondisi Data
Missing values: Terdapat missing values pada beberapa fitur seperti ArrTime, ActualElapsedTime, AirTime, TaxiIn, CarrierDelay, WeatherDelay, dll.
Duplikat: Tidak ditemukan data duplikat.
Tipe Data: Kombinasi dari data numerik kontinu, kategorikal numerik, dan objek (yang di-encode).

### Penjabaran Seluruh Fitur

| Nama Fitur           | Tipe Data | Keterangan Penggunaan                 | Deskripsi                                                                 |
|----------------------|-----------|---------------------------------------|--------------------------------------------------------------------------|
| Unnamed: 0           | int64     | Tidak digunakan                       | Indeks duplikat dari pembacaan file. Akan dihapus.                       |
| Year                 | int64     | Digunakan                             | Tahun penerbangan                                                        |
| Month                | int64     | Digunakan                             | Bulan penerbangan                                                        |
| DayofMonth           | int64     | Digunakan                             | Tanggal dalam sebulan                                                    |
| DayOfWeek            | int64     | Digunakan                             | Hari dalam minggu (1=Senin, 7=Minggu)                                     |
| DepTime              | float64   | Digunakan                             | Waktu keberangkatan aktual (HHMM)                                        |
| CRSDepTime           | int64     | Digunakan                             | Waktu keberangkatan yang dijadwalkan (HHMM)                              |
| ArrTime              | float64   | Tidak digunakan                       | Waktu kedatangan aktual                                                  |
| CRSArrTime           | int64     | Digunakan                             | Waktu kedatangan terjadwal                                               |
| UniqueCarrier        | int64     | Digunakan                             | Kode maskapai penerbangan                                                |
| FlightNum            | int64     | Tidak digunakan                       | Nomor penerbangan                                                        |
| TailNum              | int64     | Tidak digunakan                       | Nomor ekor pesawat (unik)                                                |
| ActualElapsedTime    | float64   | Tidak digunakan                       | Waktu tempuh aktual (menit)                                              |
| CRSElapsedTime       | float64   | Digunakan                             | Waktu tempuh terjadwal (menit)                                           |
| AirTime              | float64   | Tidak digunakan                       | Waktu di udara (menit)                                                   |
| ArrDelay             | float64   | Digunakan (target)                    | Keterlambatan kedatangan (menit) – akan dikonversi menjadi biner         |
| DepDelay             | float64   | Digunakan                             | Keterlambatan keberangkatan (menit)                                      |
| Origin               | int64     | Digunakan                             | Bandara asal                                                             |
| Dest                 | int64     | Digunakan                             | Bandara tujuan                                                           |
| Distance             | int64     | Digunakan                             | Jarak antar bandara (mil)                                                |
| TaxiIn               | float64   | Tidak digunakan                       | Waktu taxi masuk ke terminal (menit)                                     |
| TaxiOut              | float64   | Tidak digunakan                       | Waktu taxi keluar dari terminal (menit)                                  |
| Cancelled            | int64     | Digunakan                             | Apakah penerbangan dibatalkan (1=ya)                                     |
| CancellationCode     | int64     | Tidak digunakan                       | Alasan pembatalan                                                        |
| Diverted             | int64     | Digunakan                             | Apakah penerbangan dialihkan                                             |
| CarrierDelay         | float64   | Digunakan                             | Keterlambatan karena maskapai                                            |
| WeatherDelay         | float64   | Digunakan                             | Keterlambatan karena cuaca                                               |
| NASDelay             | float64   | Digunakan                             | Keterlambatan dari sistem kontrol NAS                                    |
| SecurityDelay        | float64   | Digunakan                             | Keterlambatan karena keamanan                                            |
| LateAircraftDelay    | float64   | Digunakan                             | Keterlambatan dari kedatangan pesawat sebelumnya                         |

> Dengan penjabaran lengkap ini, seluruh fitur pada dataset telah diidentifikasi dan diklasifikasikan apakah digunakan dalam model atau tidak, untuk menjaga transparansi dan replikasi.

---

## Data Preparation

Berikut adalah langkah-langkah *data preparation* yang dilakukan sesuai kode:

---

### 1. Pembuatan Kolom Target IsDelayed
- Untuk mengubah masalah menjadi klasifikasi biner, dibuat kolom target baru bernama IsDelayed.
- Kolom ini diturunkan dari ArrDelay dengan logika:
	- Artinya, jika keterlambatan kedatangan lebih dari 15 menit, maka IsDelayed = 1 (terlambat); jika tidak, IsDelayed = 0.
	- Langkah ini penting untuk mengubah masalah regresi menjadi klasifikasi sesuai dengan objektif model.

### 2. Encoding Fitur Kategorikal
- Semua fitur bertipe objek (kategorikal) di-*encode* menggunakan **Label Encoding**.
- Nilai kosong (`NaN`) pada kolom kategorikal diisi dengan string `'Missing'` sebelum proses encoding.
- Label Encoding dipilih agar model machine learning menerima input numerik tanpa menambah dimensi data seperti pada One-Hot Encoding.

### 3. Pemisahan Fitur dan Target
- Variabel fitur (X) dibuat dengan menghapus kolom target 'IsDelayed'.
- Variabel target (y) berisi kolom 'IsDelayed'.

### 4. Penanganan Kolom Non-Numerik yang Tersisa
- Sebelum membagi data, dicek apakah masih ada kolom non-numerik yang tersisa di X.
- Jika ada, kolom tersebut akan dihapus agar seluruh fitur menjadi numerik.

### 5. Split Data Menjadi Training dan Testing Set
- Data dibagi menjadi training set dan testing set dengan proporsi 80:20.
- Parameter random_state=42 dipakai untuk reproducibility.

### 5. Penanganan Missing Value pada Fitur Numerik (Setelah Split)
- Setelah pembagian data, dilakukan pengecekan terhadap nilai non-finite (NaN, inf, -inf) pada X_train.
- Karena terdapat nilai NaN, maka dilakukan imputasi menggunakan SimpleImputer dari Scikit-Learn.
- Strategi yang digunakan adalah mean imputation: mengganti nilai kosong dengan rata-rata kolom.


## Model Development

### Model yang Digunakan

1. Logistic Regression
Cara kerja: Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memprediksi probabilitas kelas target (dalam kasus ini, apakah sebuah penerbangan akan delay atau tidak). Model ini bekerja dengan menghitung kombinasi linier dari fitur-fitur input, lalu menerapkan fungsi aktivasi sigmoid untuk menghasilkan nilai probabilitas antara 0 dan 1. Probabilitas ini kemudian dikonversi ke dalam kelas 0 atau 1 berdasarkan ambang batas tertentu (biasanya 0.5).

- max_iter=1000: Menentukan jumlah maksimum iterasi untuk proses optimasi. Ini digunakan agar proses konvergensi lebih stabil, terutama saat jumlah fitur cukup banyak atau dataset tidak mudah dipisahkan secara linier.
- Parameter solver dan random_state tidak ditentukan secara eksplisit dalam kode, sehingga:
	- solver menggunakan nilai default yaitu 'lbfgs', yang cocok untuk dataset dengan lebih dari dua kelas dan performa baik 	untuk dataset ukuran besar.
	- random_state tidak diatur, sehingga hasilnya bisa sedikit berbeda jika dijalankan ulang karena melibatkan proses acak 	internal.

2. Random Forest Classifier
Cara kerja: Random Forest adalah algoritma ensemble yang membangun banyak decision tree selama proses pelatihan. Setiap pohon dilatih pada subset data yang berbeda (menggunakan teknik bootstrap), dan keputusan akhir diambil berdasarkan mayoritas voting dari semua pohon. Metode ini sangat efektif untuk mengurangi overfitting dan meningkatkan akurasi, terutama saat bekerja dengan dataset besar yang mengandung banyak fitur numerik dan kategorikal.

- n_estimators=100: Jumlah pohon dalam hutan. Ini adalah nilai default.
- max_depth=None: Tidak ada batasan kedalaman pohon, sehingga pohon akan berkembang hingga semua daun murni atau hingga tidak ada lagi fitur yang bisa digunakan untuk split — ini juga adalah nilai default.
- random_state: Tidak ditentukan secara eksplisit dalam kode, sehingga proses pelatihan bisa menghasilkan hasil yang sedikit berbeda setiap kali dijalankan karena elemen acak dalam pembentukan pohon.

3. Parameter Tuning dengan GridSearchCV
Untuk meningkatkan performa model Random Forest, dilakukan tuning parameter menggunakan GridSearchCV. Proses ini bertujuan mencari kombinasi parameter terbaik dengan melakukan evaluasi silang (cross-validation) pada berbagai nilai hyperparameter.

Penjelasan pemilihan parameter:
n_estimators: Menentukan jumlah pohon yang digunakan dalam ensemble. Nilai 100 dan 200 dipilih untuk membandingkan performa model ringan dan model yang lebih kompleks.
max_depth: Mengontrol kedalaman maksimum setiap pohon. Nilai 10 dan 20 dipilih untuk menghindari pohon yang terlalu dalam (overfitting), sedangkan None membiarkan pohon tumbuh bebas untuk menjadi baseline.

---

## Evaluation

### Metrik Evaluasi

- **Accuracy**: Ketepatan klasifikasi secara keseluruhan.
- **Precision**: Ketepatan model dalam memprediksi penerbangan yang benar-benar terlambat.
- **Recall**: Kemampuan model menangkap semua penerbangan yang benar-benar terlambat.
- **F1-score**: Rata-rata harmonik dari precision dan recall.
- **Confusion Matrix**: Matriks evaluasi prediksi model.

### Hasil Evaluasi

| Model              | Accuracy | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) |
|--------------------|----------|---------------|------------|--------------|---------------|------------|--------------|
| Logistic Regression | 0.95     | 0.93          | 0.94       | 0.94         | 0.97          | 0.96       | 0.96         |
| Random Forest       | 1.00     | 1.00          | 1.00       | 1.00         | 1.00          | 1.00       | 1.00         |

---

#### Logistic Regression Classification Report
          precision    recall  f1-score   support

       0       0.93      0.94      0.94    144366
       1       0.97      0.96      0.96    242986

- accuracy                           
                                0.95    387352
- macro avg 
            0.95      0.95      0.95    387352
- weighted avg 
            0.95      0.95      0.95    387352

#### Confusion Matrix - Logistic Regression

- [[136387 7979]
- [ 10625 232361]]

---

#### Random Forest Classification Report
          precision    recall  f1-score   support

       0       1.00      1.00      1.00    144366
       1       1.00      1.00      1.00    242986

- accuracy                           
                                1.00    387352
- macro avg 
            1.00      1.00      1.00    387352
- weighted avg 
            1.00      1.00      1.00    387352

#### Confusion Matrix - Random Forest

- [[144366 0]
- [ 0 242986]]

---

## Kesimpulan

- Random Forest terbukti menjadi model terbaik dengan performa sempurna pada dataset uji.
- Model ini mampu mengidentifikasi keterlambatan lebih dari 15 menit dengan akurasi dan presisi tinggi.
- Prediksi ini dapat dimanfaatkan oleh maskapai dan otoritas bandara untuk melakukan mitigasi risiko dan perbaikan layanan pelanggan.
- Seluruh tahapan, mulai dari pemilihan fitur, pembersihan data, hingga evaluasi model, dilakukan secara transparan dan berdasarkan isi notebook yang dapat direplikasi.

---
