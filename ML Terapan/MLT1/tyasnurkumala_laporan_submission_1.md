# Laporan Proyek Machine Learning - Tyas Nur Kumala

## **Domain Proyek**

### **Latar Belakang**
Kesehatan maternal merupakan salah satu aspek penting dalam meningkatkan kualitas hidup, khususnya di negara berkembang. Tingginya angka kematian ibu dan bayi akibat komplikasi kehamilan seperti preeklampsia dan diabetes gestasional masih menjadi masalah besar di banyak negara. Deteksi dini terhadap risiko kehamilan dapat membantu intervensi medis yang lebih cepat, yang pada gilirannya dapat mengurangi angka kematian tersebut.

Namun, proses deteksi risiko kehamilan seringkali dilakukan secara manual yang memerlukan keahlian medis yang spesifik dan alat kesehatan yang mahal. Di daerah pedesaan dan kurang berkembang, fasilitas medis yang terbatas menjadi tantangan besar untuk memberikan pelayanan kesehatan yang optimal. Oleh karena itu, pengembangan sistem berbasis machine learning untuk klasifikasi risiko kehamilan berbasis data medis menjadi solusi yang sangat relevan untuk meningkatkan kualitas layanan kesehatan ibu hamil.

### **Tujuan Proyek**
Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi risiko kehamilan secara otomatis menggunakan data medis ibu hamil seperti BMI, tekanan darah, dan kadar gula darah. Dengan menggunakan teknik machine learning yang tepat, sistem ini diharapkan bisa memberikan deteksi dini terhadap risiko kehamilan di daerah dengan keterbatasan fasilitas medis.

---

## **Business Understanding**

### **Problem Statements**


## 1. Bagaimana memprediksi risiko kehamilan menggunakan data klinis dan fitur kesehatan ibu?
Model deep learning, khususnya yang dibangun menggunakan TensorFlow, dapat digunakan untuk memprediksi risiko kehamilan berdasarkan data klinis dan fitur kesehatan ibu, seperti BMI, tekanan darah, kadar glukosa darah, usia, dan riwayat medis. Proses ini akan berfokus pada klasifikasi risiko kehamilan untuk membantu pengambilan keputusan medis.

## 2. Apa algoritma terbaik yang dapat digunakan untuk memodelkan prediksi risiko kehamilan?
Dalam penelitian ini, **Neural Network** berbasis TensorFlow adalah algoritma utama yang digunakan untuk memprediksi risiko kehamilan. Fokus utama dari penelitian ini adalah pada tuning hyperparameter untuk meningkatkan kinerja model dalam memprediksi risiko kehamilan.


**Goals**

Tujuan dari proyek ini adalah untuk:

1. **Memprediksi Risiko Kehamilan**: Mengembangkan model yang dapat memprediksi risiko kehamilan (Low atau High) berdasarkan data medis ibu hamil.
2. **Membandingkan Model**: Menguji dan membandingkan efektivitas beberapa model machine learning untuk menentukan model yang paling efektif dalam memprediksi risiko kehamilan.
3. **Optimasi Model**: Menggunakan **GridSearchCV** dan **RandomizedSearchCV** untuk menemukan hyperparameter terbaik yang meningkatkan akurasi model.

### **Solution Statements**
1. **GridSearchCV** dan **RandomizedSearchCV** digunakan untuk mencari hyperparameter terbaik yang dapat meningkatkan akurasi model.
2. **Neural Network** digunakan tanpa fine-tuning sebagai baseline model untuk membandingkan performa dengan model lainnya.

---

## **Data Understanding**

### **Deskripsi Dataset**
Dataset yang digunakan dalam proyek ini adalah **Maternal Health Dataset**, yang diambil dari **Kuriigram General Hospital, Bangladesh**. Dataset ini terdiri dari **1.205 record** dengan berbagai fitur medis ibu hamil yang digunakan untuk memprediksi risiko kehamilan. Dataset ini dapat diunduh dari artikel berikut:  
[**Predicting Pregnancy Complications Using Machine Learning Techniques**](https://www.sciencedirect.com/science/article/pii/S2352340925000952).

### **Variabel dalam Dataset:**
1. **Age**: Usia ibu hamil.
2. **Systolic BP**: Tekanan darah sistolik ibu.
3. **Diastolic BP**: Tekanan darah diastolik ibu.
4. **BS**: Kadar gula darah ibu.
5. **Body Temp**: Suhu tubuh ibu.
6. **BMI**: Indeks massa tubuh ibu.
7. **Previous Complications**: Riwayat komplikasi kesehatan sebelumnya.
8. **Preexisting Diabetes**: Riwayat diabetes sebelum kehamilan.
9. **Gestational Diabetes**: Keberadaan diabetes gestasional.
10. **Mental Health**: Status kesehatan mental ibu.
11. **Heart Rate**: Denyut jantung ibu.

Dataset ini digunakan untuk memprediksi dua kategori risiko kehamilan:  
- **Low**: Risiko rendah  
- **High**: Risiko tinggi

### **Masalah yang Ditemui:**
- **Missing Values**: Beberapa nilai mungkin hilang dan perlu diimputasi dengan metode yang tepat.
- **Outliers**: Beberapa fitur seperti **BMI** dan **Blood Pressure** mungkin memiliki nilai yang tidak wajar.
- **Ketidakseimbangan Kelas**: Ada kemungkinan bahwa jumlah contoh dari kelas **High** lebih sedikit dibandingkan dengan kelas **Low**, yang dapat mempengaruhi hasil model. Oleh karena itu, teknik **SMOTE (Synthetic Minority Over-sampling Technique)** akan diterapkan untuk menyeimbangkan kelas.

---

## **Data Preparation**

Data Preparation
1. Imputasi Nilai yang Hilang
Pada notebook ini, kami menggunakan dropna() untuk menghapus baris yang memiliki nilai hilang. Teknik ini dipilih karena jumlah nilai yang hilang relatif kecil dan tidak signifikan terhadap total data.

2. Penghapusan Duplikat
Kami menggunakan drop_duplicates() untuk menghapus baris yang terduplikasi dalam dataset. Langkah ini dilakukan untuk memastikan bahwa model tidak dilatih dengan data yang berulang.

3. Label Encoding
Target variabel Risk Level yang bersifat kategorikal (Low, High) diubah menjadi bentuk numerik menggunakan LabelEncoder() agar bisa digunakan dalam model machine learning.

4. Normalisasi Fitur
Fitur numerik dinormalisasi menggunakan StandardScaler agar memiliki skala yang seragam, yang penting untuk menghindari fitur dengan rentang yang lebih besar mendominasi proses pelatihan model.

5. Pembagian Data
Data dibagi menjadi 80% untuk data latih dan 20% untuk data uji menggunakan train_test_split().

6. Penanganan Ketidakseimbangan Kelas
Untuk menangani ketidakseimbangan kelas pada variabel target, digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menghasilkan data sintetis bagi kelas minoritas.


## **Model Development**

## Modeling

### 1. **Neural Network (Deep Learning)**
**Neural Network** adalah model machine learning yang terinspirasi oleh struktur otak manusia, yang terdiri dari lapisan-lapisan neuron yang saling terhubung. Model ini digunakan untuk memprediksi risiko kehamilan berdasarkan data kesehatan ibu, seperti BMI, tekanan darah, kadar glukosa darah, usia, dan riwayat medis. Berikut adalah cara kerja model Neural Network:

- **Input Layer**: Data input seperti fitur kesehatan ibu dimasukkan ke dalam lapisan pertama.
- **Hidden Layers**: Data diproses melalui beberapa lapisan tersembunyi menggunakan fungsi aktivasi (seperti ReLU) yang memungkinkan model belajar pola non-linear dalam data.
- **Output Layer**: Di lapisan terakhir, model menghasilkan output berupa klasifikasi risiko (misalnya, tinggi, sedang, rendah).
- **Pelatihan dan Pembelajaran**: Model ini dilatih dengan algoritma **backpropagation**, di mana kesalahan dihitung dan digunakan untuk memperbarui bobot pada neuron menggunakan optimisasi (misalnya Adamax).

Neural Network digunakan untuk memahami pola yang kompleks dan dapat menghasilkan prediksi yang akurat, meskipun memerlukan proses pelatihan yang lebih lama dibandingkan dengan algoritma lainnya.

### 2. **Grid Search**
**Grid Search** adalah teknik pencarian hyperparameter yang mencoba setiap kombinasi yang telah ditentukan sebelumnya dari parameter-parameter model. Metode ini memastikan bahwa model yang dihasilkan adalah yang terbaik dari semua kombinasi yang diuji. Proses ini mencakup pencarian untuk hyperparameter seperti:
- **Batch size**
- **Epochs**
- **Dropout rate**
- **Hidden units**
- **Optimizer**

Proses pencarian yang lebih eksklusif ini memberikan akurasi terbaik, tetapi memerlukan lebih banyak waktu pelatihan. Grid Search digunakan untuk menemukan parameter yang optimal guna memaksimalkan kinerja model.

### 3. **Randomized Search**
**Randomized Search** mirip dengan Grid Search, tetapi lebih efisien dalam hal waktu. Daripada mencoba semua kombinasi hyperparameter yang telah ditentukan, **Randomized Search** memilih kombinasi secara acak dari ruang hyperparameter yang ditetapkan. Dengan cara ini, Randomized Search dapat mencari hasil yang hampir setara dengan Grid Search, namun dengan waktu pelatihan yang lebih cepat. Ini sangat berguna jika pencarian parameter membutuhkan waktu yang lama atau jika ruang hyperparameter sangat besar.

### 4. **Neural Network (Tanpa Fine-Tuning)**
Model **Neural Network tanpa fine-tuning** menggunakan parameter default yang tidak diubah melalui proses pencarian hyperparameter. Model ini dilatih dengan konfigurasi standar dan dapat menghasilkan hasil yang cepat, namun akurasinya biasanya lebih rendah dibandingkan dengan model yang telah melalui proses tuning hyperparameter (seperti dengan Grid Search atau Randomized Search).

---

### **Perbandingan Model**
Berikut adalah perbandingan antara ketiga metode yang digunakan:

| **Metode**             | **Best Parameters**                                                                                                                                          | **Best Accuracy**  | **Keuntungan**                                  | **Kekurangan**                               |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------------------------------------|---------------------------------------------|
| **Grid Search**         | - Batch Size: 16 <br> - Epochs: 50 <br> - Dropout Rate: 0.3 <br> - Hidden Units: 128 <br> - Optimizer: Adamax                                                | **97.90%**         | Akurasi terbaik                                | Waktu pencarian lebih lama                  |
| **Randomized Search**   | - Optimizer: RMSprop <br> - Hidden Units: 32 <br> - Dropout Rate: 0.4 <br> - Epochs: 50 <br> - Batch Size: 32                                               | **97.81%**         | Waktu pencarian lebih cepat                    | Akurasi sedikit lebih rendah dari Grid Search |
| **Neural Network (Tanpa Fine-Tuning)** | - Menggunakan default parameter (tanpa fine-tuning)                                                                                         | **96.31%**         | Cepat dan sederhana                            | Akurasi lebih rendah tanpa fine-tuning      |

---

Dengan penjelasan ini, sudah sesuai dengan saran perbaikan yang kamu sebutkan:

1. **Penjelasan singkat tentang cara kerja Neural Network** sudah dimasukkan di bagian pertama.
2. Semua model yang disebutkan (Neural Network, Grid Search, Randomized Search) sudah dijelaskan dengan detail cara kerja dan implementasinya.
3. Semua penjelasan terkait **pemilihan algoritma**, **cara kerja**, **hyperparameter tuning**, dan **perbandingan model** telah digabungkan dalam satu kategori **Modeling**.



---

## **Evaluation**

### **Confusion Matrix**:
Confusion matrix di bawah ini menunjukkan perbandingan antara **nilai aktual** dan **prediksi model** untuk dua kelas risiko kehamilan (Low dan High).

| Actual / Predicted | 0 (Low) | 1 (High) |
|--------------------|---------|----------|
| **0 (Low)**        | 82      | 3        |
| **1 (High)**       | 4       | 128      |

### **Penjelasan Confusion Matrix**:
- **True Positives (TP)**: 128 (Model benar memprediksi kehamilan dengan risiko tinggi).
- **True Negatives (TN)**: 82 (Model benar memprediksi kehamilan dengan risiko rendah).
- **False Positives (FP)**: 3 (Model salah memprediksi kehamilan dengan risiko tinggi sebagai risiko rendah).
- **False Negatives (FN)**: 4 (Model salah memprediksi kehamilan dengan risiko rendah sebagai risiko tinggi).

### **Metrik Evaluasi**:
Berdasarkan confusion matrix, berikut adalah metrik evaluasi model:

| **Metric**  | **Precision** | **Recall** | **F1-Score** | **Support** |
|-------------|---------------|------------|--------------|-------------|
| **0 (Low)** | 0.95          | 0.96       | 0.96         | 85          |
| **1 (High)**| 0.98          | 0.97       | 0.97         | 132         |

- **Accuracy**: 0.97 (Akurasi model pada data uji adalah 97%).
- **Macro Average**: Precision, Recall, dan F1-Score untuk kedua kelas (Low dan High) adalah **0.97**, yang menunjukkan keseimbangan antara keduanya.
- **Weighted Average**: Precision, Recall, dan F1-Score untuk seluruh model adalah **0.97**, menandakan model bekerja sangat baik pada dataset yang seimbang.

### **Kesimpulan**:
- Model ini menunjukkan **performansi yang sangat baik**, dengan **akurasi** mencapai **97%**.
- **Precision** dan **Recall** untuk kedua kelas (**Low** dan **High**) sangat tinggi, dengan model lebih cenderung memprediksi **risiko tinggi (High)** dengan akurat (precision = 0.98, recall = 0.97).
- **Confusion Matrix** menunjukkan bahwa **false positives** dan **false negatives** relatif sedikit, mengindikasikan bahwa model sangat efektif dalam mengklasifikasikan **risiko kehamilan**.

---

## **Perbandingan Antara Grid Search, Randomized Search, dan Neural Network (Tanpa Fine-Tuning)**

### **1. Apa yang Dapat Dipelajari dari Perbandingan Tiga Metode?**
Dalam penelitian ini, tiga pendekatan berbeda digunakan untuk melatih model klasifikasi risiko kehamilan: **Grid Search**, **Randomized Search**, dan **Neural Network tanpa fine-tuning**. Berikut adalah perbandingan dari masing-masing pendekatan.

| **Metode**             | **Best Parameters**                                                                                                                                          | **Best Accuracy**  | **Keuntungan**                                  | **Kekurangan**                               |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------------------------------------|---------------------------------------------|
| **Grid Search**         | - Batch Size: 16 <br> - Epochs: 50 <br> - Dropout Rate: 0.3 <br> - Hidden Units: 128 <br> - Optimizer: Adamax                                                | **97.90%**         | Akurasi terbaik                                | Waktu pencarian lebih lama                  |
| **Randomized Search**   | - Optimizer: RMSprop <br> - Hidden Units: 32 <br> - Dropout Rate: 0.4 <br> - Epochs: 50 <br> - Batch Size: 32                                               | **97.81%**         | Waktu pencarian lebih cepat                    | Akurasi sedikit lebih rendah dari Grid Search |
| **Neural Network (Tanpa Fine-Tuning)** | - Menggunakan default parameter (tanpa fine-tuning)                                                                                         | **96.31%**         | Cepat dan sederhana                            | Akurasi lebih rendah tanpa fine-tuning      |

### **2. Apa Perbedaan Utama Antara Ketiga Metode Ini?**
- **Grid Search** memberikan **akurasi tertinggi** (97.90%) karena mencoba semua kombinasi hyperparameter yang telah ditentukan, tetapi memerlukan waktu yang lebih lama.
- **Randomized Search** memberikan **akurasi yang hampir sama** (97.81%) dengan **Grid Search** namun lebih efisien dalam hal waktu pencarian.
- **Neural Network tanpa fine-tuning** hanya menggunakan **default parameter**, menghasilkan **akurasi yang lebih rendah** (96.31%) tetapi lebih cepat.

### **Kesimpulan**:
- **Grid Search** memberikan hasil **akurasi tertinggi** (97.90%) tetapi memakan waktu lebih lama.
- **Randomized Search** memberikan hasil yang hampir **sama baiknya** (97.81%) dengan **Grid Search**, tetapi jauh lebih cepat.
- **Neural Network tanpa fine-tuning** memberikan hasil **akurasi lebih rendah** (96.31%) tetapi lebih **sederhana** dan cepat.
- **Penggunaan pencarian hyperparameter** (**Grid Search** dan **Randomized Search**) jelas memberikan **keuntungan dalam akurasi model**, meskipun **Randomized Search** lebih efisien dalam **waktu pencarian**.

---

Laporan ini telah dilengkapi dengan bagian **Model Development**, **Evaluation**, dan **Perbandingan Antara Grid Search, Randomized Search, dan Neural Network**. Semuanya sudah disesuaikan dengan format Markdown. Jika ada tambahan atau revisi lain yang diperlukan, beri tahu saya!
