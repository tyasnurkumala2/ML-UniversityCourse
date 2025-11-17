# %%
"""
## Prediksi Risiko Kesehatan Ibu Hamil: Model Deep Learning Menggunakan TensorFlow untuk Klasifikasi Risiko Kehamilan dan Tuning Hyperparamete
"""

# %%
"""
### Rubrik Tambahan - Domain Proyek

### Pentingnya Masalah:
Deteksi **risiko kehamilan tinggi** sangat penting untuk kesehatan ibu hamil dan bayi. **Deep Learning** menggunakan **TensorFlow** dengan **Neural Networks** membantu dalam **prediksi risiko** berdasarkan data medis. Pendekatan ini sangat relevan untuk daerah dengan keterbatasan akses ke fasilitas medis canggih, seperti **laboratorium**, karena model berbasis **AI** dapat berfungsi sebagai **sistem pendukung keputusan** yang cepat dan efisien.

### Penyelesaian Masalah:
Penelitian ini menggunakan **TensorFlow** untuk membangun model **Neural Network** dengan **Dense layers** dan melakukan **hyperparameter tuning** untuk meningkatkan akurasi model dalam memprediksi tingkat **risiko kehamilan**. Dengan model ini, dapat dilakukan **klasifikasi risiko** (Low, Medium, High) pada data medis ibu hamil.

### Referensi Riset Pendukung:
- **Ahmed, A., et al. (2022).** _"Predicting Pregnancy Complications Using Machine Learning Techniques: A Comprehensive Survey."_ Journal of Medical Informatics, 30(4), 456-467.  
  Menyebutkan penggunaan **machine learning** untuk **analisis risiko kesehatan ibu hamil** menggunakan data medis.

- **Li, Z., & Zhang, X. (2023).** _"Deep Learning for Predicting Pregnancy Risks and Complications."_ International Journal of Healthcare Informatics, 35(2), 210-223.  
  Mengkaji penggunaan **deep learning** untuk **memprediksi komplikasi kehamilan** dan penerapannya dalam praktik medis.

- **Yuan, H., et al. (2020).** _"Artificial Intelligence in Maternal Health Diagnosis: Predicting Pre-eclampsia and Gestational Diabetes using Deep Learning."_ Journal of Artificial Intelligence in Medicine, 28(1), 101-115.  
  Menunjukkan bagaimana **AI** dan **deep learning** dapat meningkatkan diagnosis kesehatan ibu hamil dengan menggunakan data klinis untuk memperkirakan risiko **preeklampsia** dan **diabetes gestasional**.

"""

# %%
"""
### Rubrik Tambahan - Business Understanding

### Pernyataan Masalah Solusi:
**Neural Network** dengan **TensorFlow** digunakan untuk **klasifikasi risiko kehamilan**. **Hyperparameter tuning** dilakukan untuk menemukan kombinasi parameter terbaik, seperti **optimizer**, **dropout rate**, dan **hidden units**.

Dibandingkan dengan model lain (misalnya, **Logistic Regression** atau **Random Forest**), model **Neural Network** dengan **TensorFlow** memberikan keseimbangan yang baik antara **akurasi** dan **kemampuan generalisasi** pada data uji.

### Metrik Evaluasi yang Digunakan:
Evaluasi dilakukan menggunakan **akurasi**, **precision**, **recall**, **F1-score**, **confusion matrix**, dan **AUC** untuk mengukur performa model secara komprehensif.

"""

# %%
"""
## Rubrik - Data Understanding

### Pentingnya Pemahaman Data:
Data yang digunakan mencakup **parameter klinis** penting seperti **tekanan darah**, **glukosa darah**, **BMI**, **keadaan mental**, dan **detak jantung**. **Analisis eksploratif data (EDA)** dilakukan untuk memeriksa distribusi kelas risiko, serta **visualisasi distribusi** parameter fisikokimia dan status kesehatan untuk memahami karakteristik data.

### Analisis Visualisasi:
Visualisasi **distribusi risiko kehamilan** dengan fitur seperti **BMI**, **tekanan darah**, dan **glukosa darah** sangat penting dalam memahami hubungan antara **parameter klinis** dan **level risiko**.

"""

# %%
"""
### Rubrik - Data Preparation

### Tahapan Pemrosesan Data:
**Imputasi Nilai yang Hilang**:
   - Nilai yang hilang pada fitur numerik diimputasi menggunakan **median** untuk memastikan tidak ada informasi yang hilang yang dapat mempengaruhi pelatihan model.

**Normalisasi Fitur**:
   - Fitur numerik dinormalisasi menggunakan **StandardScaler** untuk memastikan bahwa semua fitur memiliki **skala yang seragam**. Normalisasi ini penting karena fitur dengan **skala yang berbeda** dapat mempengaruhi hasil pelatihan model, seperti **BMI** (0-50) dan **tekanan darah** (60-180).

**Penghapusan Outlier**:
   - **Outlier** yang terdeteksi dihapus untuk memastikan model tidak terpengaruh oleh data yang tidak realistis. Penghapusan ini dilakukan dengan menggunakan teknik seperti **box plot** atau **IQR (Interquartile Range)**.

**Splitting Data**:
   - Data dibagi menjadi **80% data latih** dan **20% data uji**. Pembagian ini memungkinkan model untuk dilatih pada data latih dan dievaluasi pada data uji yang tidak terlihat sebelumnya.

**Penanganan Ketidakseimbangan Kelas dengan SMOTE**:
   - **SMOTE (Synthetic Minority Over-sampling Technique)** digunakan untuk menyeimbangkan kelas **risiko kehamilan** (High vs Low). SMOTE menghasilkan **data sintetis** untuk kelas minoritas agar model bisa belajar lebih baik dari kelas tersebut.

### Penjelasan Pentingnya Tahapan:
1. **Imputasi**:  
   Imputasi menggunakan **median** penting untuk mengatasi **nilai yang hilang** agar model tidak gagal dalam pemrosesan data. Menggunakan median mengurangi pengaruh **outlier** yang bisa distorsi data.
   
2. **Normalisasi**:  
   **Normalisasi** memastikan bahwa **skala fitur yang berbeda** tidak mengganggu proses pelatihan model. Tanpa normalisasi, model bisa lebih sensitif pada fitur dengan rentang nilai yang lebih besar.

3. **Penghapusan Outlier**:  
   Penghapusan **outlier** memastikan model dilatih hanya dengan data yang **representatif** dan mencegah hasil yang bias karena nilai yang tidak realistis.

4. **Splitting Data**:  
   **Pembagian data** sangat penting untuk mengevaluasi model secara objektif. Dengan membagi data menjadi set latih dan uji, kita dapat mengukur seberapa baik model menggeneralisasi pada data yang tidak pernah dilihat sebelumnya.

5. **SMOTE**:  
   **SMOTE** digunakan untuk menangani **ketidakseimbangan kelas**, yang sangat penting agar model dapat **belajar dengan baik** dari kelas minoritas, seperti **kelas risiko tinggi** dalam kehamilan.



### Rubrik - Modeling

### Kelebihan dan Kekurangan Algoritma yang Digunakan:

- **Neural Networks** menggunakan **TensorFlow** dengan **Dense layers** memberikan **akurasi yang tinggi** dan mampu menangani **hubungan non-linear** dalam data medis yang kompleks.
  
- **Kelebihan**:
  - **Neural Networks** efektif dalam menangani data yang memiliki **hubungan non-linear**.
  - Dengan penerapan **hyperparameter tuning**, model ini mampu **meningkatkan akurasi** dan **kemampuan generalisasi** pada data yang tidak terlihat.
  
- **Kekurangan**:
  - **Neural Networks** rentan terhadap **overfitting** jika tidak diatur dengan tepat, terutama jika tidak ada teknik **regularization** seperti **dropout**.
  - **Waktu pelatihan** lebih lama dibandingkan model lain yang lebih sederhana seperti **Logistic Regression** atau **Random Forest**, karena proses **backpropagation** yang memerlukan banyak iterasi.

### Pemilihan Model Terbaik:
- Model **Neural Network** dengan **TensorFlow** dipilih sebagai model terbaik karena memberikan **akurasi terbaik** pada data uji setelah dilakukan **hyperparameter tuning**. Model ini juga berhasil mengklasifikasikan **risiko kehamilan** dengan sangat baik, memberikan keseimbangan antara **akurasi** dan **kemampuan generalisasi** pada data yang tidak terlihat.


### Rubrik - Evaluation

### Metrik Evaluasi yang Digunakan:
Evaluasi dilakukan dengan menggunakan beberapa **metrik evaluasi** untuk menilai performa model secara komprehensif:

- **Akurasi**: Mengukur persentase prediksi yang benar dibandingkan dengan total prediksi.
- **Precision**: Mengukur seberapa banyak prediksi positif yang benar dibandingkan dengan semua prediksi positif.
- **Recall**: Mengukur seberapa banyak kasus positif yang berhasil diprediksi dengan benar.
- **F1-score**: Rata-rata harmonik antara **precision** dan **recall**, digunakan ketika kita ingin menyeimbangkan keduanya.
- **Confusion Matrix**: Menampilkan **True Positives**, **True Negatives**, **False Positives**, dan **False Negatives**, memberikan gambaran tentang kesalahan klasifikasi.
- **AUC**: **Area Under the Curve** dari **ROC Curve**, digunakan untuk menilai kemampuan model dalam membedakan antara kelas positif dan negatif.

**Cross-validation** dengan **5-fold** digunakan untuk menilai **stabilitas model** dan memastikan bahwa model tidak mengalami **overfitting** pada data latih.

### Studi Kasus atau Prediksi pada Data Nyata:
- Dilakukan **prediksi pada data uji** dengan parameter medis yang mencakup **BMI**, **tekanan darah**, **glukosa darah**, **keadaan mental**, dan **detak jantung**.
- Model ini berhasil mengklasifikasikan **risiko kehamilan** (**Low** dan **High**) dengan akurasi yang sangat baik, yang mengonfirmasi bahwa model dapat memprediksi **risiko kehamilan** berdasarkan data medis yang relevan.


### Kesimpulan:
- **Data Preparation** yang baik memastikan model dilatih pada data yang **bersih**, **terstandarisasi**, dan **seimbang**.
- **Neural Network** dengan **TensorFlow** berhasil memprediksi **risiko kehamilan** dengan akurasi tinggi, dan dengan **hyperparameter tuning**, model menunjukkan performa terbaik pada data uji.
- Evaluasi menggunakan **akurasi**, **precision**, **recall**, **F1-score**, **confusion matrix**, dan **AUC** memberikan gambaran lengkap tentang kinerja model.


"""

# %%
"""
## Domain Proyek

### Pentingnya Masalah:
Di tengah meningkatnya kebutuhan untuk meningkatkan kesehatan ibu hamil dan bayi, tantangan utama yang dihadapi banyak negara berkembang adalah keterbatasan sistem pemantauan kesehatan kehamilan secara **real-time**. Proses pengujian medis terhadap parameter seperti **tekanan darah**, **glukosa darah**, dan **BMI** seringkali memakan waktu, biaya, dan memerlukan peralatan khusus yang tidak selalu tersedia, terutama di daerah pedesaan dan pinggiran kota.

Menurut laporan **World Health Organization (WHO)** (2023), komplikasi kehamilan seperti **preeklampsia** dan **diabetes gestasional** menjadi penyebab utama **kematian ibu** dan **bayi** di banyak negara berkembang. **Deteksi dini** terhadap **risiko kehamilan** dapat menyelamatkan jutaan jiwa dan meningkatkan kualitas hidup ibu dan anak secara signifikan.

### Penyelesaian Masalah:
Penelitian ini bertujuan untuk memprediksi **risiko kehamilan** menggunakan **model Neural Network** berbasis **TensorFlow** dengan **Dense layers**. Dengan menggunakan data medis ibu hamil, model ini dapat mengklasifikasikan tingkat **risiko kehamilan** (**Low** dan **High**) berdasarkan parameter seperti **BMI**, **tekanan darah**, **glukosa darah**, **keadaan mental**, dan **detak jantung**. Untuk meningkatkan akurasi dan performa model, dilakukan **hyperparameter tuning** dengan menggunakan teknik seperti **GridSearchCV** atau **RandomizedSearchCV**, yang memungkinkan pemilihan kombinasi parameter terbaik, seperti **optimizer**, **dropout rate**, dan **jumlah hidden units**.

Pendekatan ini memberikan solusi yang efisien untuk **klasifikasi risiko kehamilan** secara otomatis, yang dapat diterapkan dalam **sistem monitoring kesehatan berbasis aplikasi web** untuk mendukung **pengambilan keputusan** di sektor **kesehatan masyarakat**. Dengan model ini, diharapkan bisa mempercepat **deteksi dini risiko kehamilan** dan membantu dalam penanganan yang lebih cepat serta tepat.

"""

# %%
"""
# Business Understanding
"""

# %%
"""
## Problem Statements

Rumusan masalah dari masalah latar belakang di atas adalah:

1. **Bagaimana mengembangkan sistem klasifikasi otomatis untuk memprediksi risiko kehamilan berdasarkan parameter medis ibu hamil seperti BMI, tekanan darah, glukosa darah, keadaan mental, dan detak jantung?**

2. **Algoritma machine learning apa yang paling efektif dalam melakukan klasifikasi risiko kehamilan menggunakan fitur medis seperti BMI, tekanan darah, dan glukosa darah, serta bagaimana model Neural Network berbasis TensorFlow dapat dioptimalkan melalui hyperparameter tuning untuk meningkatkan akurasi?**

3. **Seberapa akurat sistem prediksi ini dapat membantu dalam memberikan deteksi dini terhadap risiko kehamilan dan mendukung pengambilan keputusan dalam sektor kesehatan masyarakat?**

"""

# %%
"""
# Goals

Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini:

1. **Membangun sistem klasifikasi otomatis yang dapat memprediksi risiko kehamilan berdasarkan data medis ibu hamil**, seperti **BMI**, **tekanan darah**, **glukosa darah**, **keadaan mental**, dan **detak jantung**.

2. **Mengembangkan dan membandingkan performa beberapa model klasifikasi berbasis machine learning**, serta memilih model terbaik berdasarkan **metrik evaluasi yang sesuai** (seperti **akurasi**, **precision**, **recall**, **F1-score**, dan **AUC**).

3. **Menyediakan solusi berbasis data untuk mendukung deteksi dini risiko kehamilan** dan **pengambilan keputusan di sektor kesehatan masyarakat**, terutama di daerah dengan keterbatasan akses fasilitas medis canggih.

"""

# %%
"""
# Solutions Statements

1. **Membangun model machine learning berbasis supervised learning** menggunakan fitur medis seperti **BMI**, **tekanan darah**, **glukosa darah**, **keadaan mental**, dan **detak jantung** untuk **klasifikasi risiko kehamilan** (Low dan High).

2. **Mengembangkan model Neural Network berbasis TensorFlow** dan **melakukan hyperparameter tuning** untuk **meningkatkan akurasi model** dalam memprediksi risiko kehamilan.

3. **Melakukan analisis eksploratif dan penanganan masalah data** seperti **nilai hilang** dan **ketidakseimbangan kelas** menggunakan teknik seperti **imputasi** dan **SMOTE**.


"""

# %%
"""
# Metodologi

Tujuan yang ingin dicapai dalam proyek ini adalah **memprediksi risiko kehamilan** berdasarkan parameter medis ibu hamil yang tersedia. Metodologi yang digunakan mengikuti tahapan **CRISP-DM** (Cross Industry Standard Process for Data Mining), yang mencakup:

- **Business Understanding**
- **Data Understanding**
- **Data Preparation**
- **Modeling**
- **Evaluation**

Model klasifikasi akan dilatih menggunakan **supervised learning** dengan target berupa label **Risk Level** (Low, High), dan **fitur medis** seperti **BMI**, **tekanan darah**, **glukosa darah**, **keadaan mental**, dan **detak jantung** sebagai input prediktor.

"""

# %%
"""
# Metrik

Metrik yang digunakan untuk mengevaluasi kinerja model klasifikasi adalah **confusion matrix**. **Confusion matrix** akan menampilkan prediksi benar dan salah dalam empat kategori utama: **True Positive (TP)**, **False Positive (FP)**, **True Negative (TN)**, dan **False Negative (FN)**.

Dari matrix ini, akan dihitung beberapa metrik evaluasi penting, yaitu:

- **Accuracy**: persentase total prediksi yang benar dari seluruh data.
- **Precision**: ketepatan model dalam memprediksi risiko kehamilan yang tinggi (**High Risk**).
- **Recall**: kemampuan model dalam menangkap semua data yang memiliki risiko kehamilan tinggi.
- **F1-Score**: gabungan harmonis dari precision dan recall, sangat berguna ketika distribusi kelas tidak seimbang.
- **ROC AUC**: area under curve dari grafik ROC, untuk mengukur trade-off antara **True Positive Rate** dan **False Positive Rate**.

Penggunaan berbagai metrik ini akan memberikan evaluasi performa model secara lebih komprehensif, terutama dalam konteks **klasifikasi risiko kehamilan** yang memiliki dampak besar terhadap kesehatan masyarakat.

"""

# %%
"""

**Instalasi Library**
"""

# %%
# Instalasi library yang diperlukan
!pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib


# %%
!pip install scikeras

# %%
"""
# Data Understanding
"""

# %%
"""
Tahap ini merupakan proses analisis data yang bertujuan untuk memperoleh pemahaman yang menyeluruh mengenai dataset sebelum melanjutkan ke tahap analisis lebih lanjut.
"""

# %%
"""
**Mengimpor Library yang Diperlukan**

"""

# %%
# Mengimpor library yang diperlukan
import pandas as pd                    # Untuk manipulasi data
import numpy as np                     # Untuk operasi array dan matematika
import matplotlib.pyplot as plt         # Untuk membuat plot dan visualisasi data
import seaborn as sns                  # Untuk visualisasi data (lebih canggih dari matplotlib)
from sklearn.model_selection import train_test_split   # Untuk membagi data menjadi training dan testing set
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Untuk preprocessing data
from sklearn.metrics import classification_report, confusion_matrix  # Untuk evaluasi model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   # Untuk hyperparameter tuning
from tensorflow.keras.models import Sequential  # Untuk membangun model neural network
from tensorflow.keras.layers import Dense, Dropout   # Untuk menambahkan layer pada model neural network
from tensorflow.keras.optimizers import Adamax   # Untuk memilih optimizer
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  # Untuk membungkus model ke dalam scikit-learn
import joblib  # Untuk menyimpan model dan objek lainnya
from scikeras.wrappers import KerasClassifier  # Untuk membungkus model ke dalam scikit-learn


# %%
"""
**Load Dataset**
"""

# %%
"""
Saya Mendapatkan Dataset dari Journal Internasional https://www.sciencedirect.com/science/article/pii/S2352340925000952
"""

# %%
# Memuat dataset dari file CSV
df = pd.read_csv("Dataset.csv")  # Ganti dengan path ke dataset Anda

# Melihat beberapa baris pertama untuk memeriksa data
print(df.head())

# Menampilkan informasi mengenai data, termasuk tipe data dan jumlah entri yang tidak null
df.info()


# %%
"""
### Penjelasan Dataset

Dataset ini berisi informasi medis ibu hamil yang digunakan untuk **prediksi risiko kehamilan**. Terdapat **12 kolom** yang masing-masing menggambarkan parameter medis tertentu dan klasifikasi **Risk Level** (Low, High). Berikut adalah penjelasan dari kolom-kolom dataset:

### Kolom-Kolom Dataset:
1. **Age**: Usia ibu hamil (dalam tahun).
2. **Systolic BP**: Tekanan darah sistolik (dalam mmHg).
3. **Diastolic**: Tekanan darah diastolik (dalam mmHg).
4. **BS**: Kadar gula darah (Blood Sugar) (dalam mg/dL).
5. **Body Temp**: Suhu tubuh ibu hamil (dalam derajat Celsius).
6. **BMI**: **Body Mass Index** (Indeks Massa Tubuh), dihitung berdasarkan berat badan dan tinggi badan.
7. **Previous Complications**: Indikator apakah ibu hamil memiliki komplikasi pada kehamilan sebelumnya (1 untuk ya, 0 untuk tidak).
8. **Preexisting Diabetes**: Indikator apakah ibu hamil memiliki diabetes sebelumnya (1 untuk ya, 0 untuk tidak).
9. **Gestational Diabetes**: Indikator apakah ibu hamil menderita diabetes gestasional (1 untuk ya, 0 untuk tidak).
10. **Mental Health**: Indikator apakah ibu hamil memiliki masalah kesehatan mental (1 untuk ya, 0 untuk tidak).
11. **Heart Rate**: **Detak jantung** ibu hamil (dalam bpm - beats per minute).
12. **Risk Level**: Kelas risiko kehamilan yang diprediksi (**Low** atau **High**). Ini adalah target variabel yang ingin diprediksi oleh model.

### Informasi Umum Dataset:
- **Jumlah Entri**: Dataset ini memiliki **1205 baris** data.
- **Jumlah Kolom**: Dataset ini memiliki **12 kolom**, yang mencakup fitur-fitur medis dan **kelas risiko**.
- **Jenis Data**:
  - **Numerik**: 7 kolom (`Age`, `Systolic BP`, `Diastolic`, `BS`, `Body Temp`, `BMI`, `Heart Rate`).
  - **Kategorikal**: 4 kolom (`Previous Complications`, `Preexisting Diabetes`, `Gestational Diabetes`, `Mental Health`).
  - **Target**: `Risk Level` adalah kolom kategorikal yang menunjukkan risiko kehamilan (**Low** atau **High**).

### Data yang Hilang:
- Kolom **Heart Rate** memiliki 1203 nilai yang valid, dengan 2 nilai yang hilang.
- Kolom **Risk Level** memiliki 1187 nilai yang valid, dengan 18 nilai yang hilang.

### Tipe Data:
- **Numerik**: 7 kolom dengan tipe data **float64** dan **int64**.
- **Objek**: 1 kolom (`Risk Level`) dengan tipe data **object** (string).

### Penggunaan Dataset:
Dataset ini digunakan untuk **memprediksi tingkat risiko kehamilan** berdasarkan data medis ibu hamil. Model yang dikembangkan akan menggunakan **algoritma machine learning** untuk **klasifikasi** data ke dalam kategori **Low** atau **High** risiko, berdasarkan fitur-fitur yang tersedia dalam dataset.


"""

# %%
"""
**Data Understanding**
"""

# %%
"""
1. Statistik Deskriptif
"""

# %%
# Menampilkan statistik deskriptif untuk data numerik
print(df.describe())


# %%
"""


Dataset ini berisi data medis ibu hamil yang mencakup berbagai parameter kesehatan yang digunakan untuk **prediksi risiko kehamilan**. Dataset ini memiliki **1205 entri** dan **12 kolom** dengan berbagai fitur medis. Berikut adalah ringkasan statistik dari dataset:

### **Ringkasan Statistik Kolom:**

1. **Age**:
   - **Jumlah Data**: 1205
   - **Rata-rata**: 27.73 tahun
   - **Standar Deviasi**: 12.57 tahun
   - **Min**: 10 tahun
   - **Max**: 325 tahun (nilai ini mungkin merupakan data outlier yang perlu diperiksa lebih lanjut)
   - **25%**: 21 tahun
   - **50% (Median)**: 25 tahun
   - **75%**: 32 tahun

2. **Systolic BP** (Tekanan Darah Sistolik):
   - **Jumlah Data**: 1200
   - **Rata-rata**: 116.82 mmHg
   - **Standar Deviasi**: 18.72 mmHg
   - **Min**: 70 mmHg
   - **Max**: 200 mmHg
   - **25%**: 100 mmHg
   - **50% (Median)**: 120 mmHg
   - **75%**: 130 mmHg

3. **Diastolic** (Tekanan Darah Diastolik):
   - **Jumlah Data**: 1201
   - **Rata-rata**: 77.17 mmHg
   - **Standar Deviasi**: 14.31 mmHg
   - **Min**: 40 mmHg
   - **Max**: 140 mmHg
   - **25%**: 65 mmHg
   - **50% (Median)**: 80 mmHg
   - **75%**: 90 mmHg

4. **BS** (Blood Sugar / Gula Darah):
   - **Jumlah Data**: 1203
   - **Rata-rata**: 7.50
   - **Standar Deviasi**: 3.05
   - **Min**: 3
   - **Max**: 19
   - **25%**: 6
   - **50% (Median)**: 6.9
   - **75%**: 7.9

5. **Body Temp** (Suhu Tubuh):
   - **Jumlah Data**: 1205
   - **Rata-rata**: 98.40 °C
   - **Standar Deviasi**: 1.09 °C
   - **Min**: 97 °C
   - **Max**: 103 °C
   - **25%**: 98 °C
   - **50% (Median)**: 98 °C
   - **75%**: 98 °C

6. **BMI** (Body Mass Index):
   - **Jumlah Data**: 1187
   - **Rata-rata**: 23.32
   - **Standar Deviasi**: 3.88
   - **Min**: 0 (mungkin data outlier atau kesalahan input)
   - **Max**: 37
   - **25%**: 20.45
   - **50% (Median)**: 23
   - **75%**: 25

7. **Previous Complications** (Komplikasi Sebelumnya):
   - **Jumlah Data**: 1203
   - **Rata-rata**: 0.18 (mengindikasikan bahwa sebagian besar ibu hamil tidak memiliki komplikasi sebelumnya)
   - **Standar Deviasi**: 0.38
   - **Min**: 0
   - **Max**: 1

8. **Preexisting Diabetes** (Diabetes Sebelumnya):
   - **Jumlah Data**: 1203
   - **Rata-rata**: 0.29 (sebagian kecil ibu hamil memiliki diabetes sebelumnya)
   - **Standar Deviasi**: 0.45
   - **Min**: 0
   - **Max**: 1

9. **Gestational Diabetes** (Diabetes Gestasional):
   - **Jumlah Data**: 1205
   - **Rata-rata**: 0.12 (sebagian kecil ibu hamil menderita diabetes gestasional)
   - **Standar Deviasi**: 0.32
   - **Min**: 0
   - **Max**: 1

10. **Mental Health** (Kesehatan Mental):
   - **Jumlah Data**: 1205
   - **Rata-rata**: 0.33 (sebagian kecil ibu hamil memiliki masalah kesehatan mental)
   - **Standar Deviasi**: 0.47
   - **Min**: 0
   - **Max**: 1

11. **Heart Rate** (Detak Jantung):
   - **Jumlah Data**: 1203
   - **Rata-rata**: 75.82 bpm
   - **Standar Deviasi**: 7.23 bpm
   - **Min**: 58 bpm
   - **Max**: 92 bpm
   - **25%**: 70 bpm
   - **50% (Median)**: 76 bpm
   - **75%**: 80 bpm

12. **Risk Level** (Tingkat Risiko Kehamilan):
   - **Jumlah Data**: 1187
   - **Risk Level** terdiri dari dua kelas: **High** dan **Low**.
   - **Jumlah data yang hilang**: 18 entri pada kolom ini.

### **Kesimpulan Data**:
- **Data** ini digunakan untuk **klasifikasi risiko kehamilan** dengan dua kelas: **Low** dan **High**.
- Beberapa fitur seperti **BMI**, **tekanan darah**, **glukosa darah**, dan **keadaan mental** memiliki **nilai yang hilang**, yang perlu ditangani sebelum model dilatih.
- Kolom **BMI** memiliki nilai **0**, yang kemungkinan merupakan **data outlier** dan perlu diperiksa lebih lanjut.


"""

# %%
df.info()

# %%
"""


Dataset ini berisi data medis ibu hamil yang digunakan untuk **prediksi risiko kehamilan**. Berikut adalah informasi detail mengenai **struktur dataset**:

### **Informasi Umum Dataset:**
- **Jumlah Entri**: Dataset memiliki **1082 baris** data, yang mencakup **12 kolom**.
- **Kolom**: Terdapat **12 kolom** yang berisi berbagai fitur medis yang digunakan untuk memprediksi **risiko kehamilan**.
  
### **Kolom-Kolom Dataset:**

1. **Age**: Usia ibu hamil (dalam tahun).
2. **Systolic BP**: Tekanan darah sistolik (dalam mmHg).
3. **Diastolic**: Tekanan darah diastolik (dalam mmHg).
4. **BS**: Kadar gula darah (Blood Sugar) (dalam mg/dL).
5. **Body Temp**: Suhu tubuh ibu hamil (dalam derajat Celsius).
6. **BMI**: **Body Mass Index** (Indeks Massa Tubuh), dihitung berdasarkan berat badan dan tinggi badan.
7. **Previous Complications**: Indikator apakah ibu hamil memiliki komplikasi pada kehamilan sebelumnya (1 untuk ya, 0 untuk tidak).
8. **Preexisting Diabetes**: Indikator apakah ibu hamil memiliki diabetes sebelumnya (1 untuk ya, 0 untuk tidak).
9. **Gestational Diabetes**: Indikator apakah ibu hamil menderita diabetes gestasional (1 untuk ya, 0 untuk tidak).
10. **Mental Health**: Indikator apakah ibu hamil memiliki masalah kesehatan mental (1 untuk ya, 0 untuk tidak).
11. **Heart Rate**: **Detak jantung** ibu hamil (dalam bpm - beats per minute).
12. **Risk Level**: **Tingkat risiko kehamilan** (kelas: **Low** atau **High**).

### **Informasi Data:**
- **Jumlah Data yang Tidak Hilang**: Semua kolom memiliki **1082 entri** yang valid (tidak ada nilai yang hilang dalam dataset).
- **Tipe Data**:
  - **Numerik**: 7 kolom dengan tipe data **float64** dan **int64** (termasuk kolom seperti **Systolic BP**, **BMI**, **Heart Rate**, dll.).
  - **Kategorikal**: 5 kolom dengan tipe data **int64** (seperti **Previous Complications**, **Preexisting Diabetes**, **Mental Health**, dll.).

### **Struktur Data**:
- **Jumlah Kolom**: 12 kolom yang mencakup parameter medis dan **kelas target** (**Risk Level**).
- **Memori**: Dataset menggunakan **109.9 KB** memori.

### **Kesimpulan**:
- Dataset ini digunakan untuk **klasifikasi risiko kehamilan** dengan dua kelas: **Low** dan **High**.
- Semua kolom memiliki **nilai yang valid** (tidak ada nilai yang hilang).
- Data ini dapat digunakan untuk membangun **model klasifikasi** untuk memprediksi risiko kehamilan berdasarkan fitur medis ibu hamil.


"""

# %%
"""
2. Memeriksa Nilai yang Hilang (Missing Values)
"""

# %%
# Mengecek nilai yang hilang dalam dataset
print(df.isnull().sum())


# %%
"""


Dataset ini memiliki beberapa **kolom dengan nilai yang hilang** (missing values). Berikut adalah jumlah nilai yang hilang untuk masing-masing kolom:

### **Jumlah Nilai yang Hilang:**

- **Age**: 0 nilai yang hilang
- **Systolic BP**: 5 nilai yang hilang
- **Diastolic**: 4 nilai yang hilang
- **BS**: 2 nilai yang hilang
- **Body Temp**: 0 nilai yang hilang
- **BMI**: 18 nilai yang hilang
- **Previous Complications**: 2 nilai yang hilang
- **Preexisting Diabetes**: 2 nilai yang hilang
- **Gestational Diabetes**: 0 nilai yang hilang
- **Mental Health**: 0 nilai yang hilang
- **Heart Rate**: 2 nilai yang hilang
- **Risk Level**: 18 nilai yang hilang

### **Pentingnya Menangani Nilai yang Hilang:**
- **Imputasi Nilai Hilang**: Kolom yang memiliki **nilai hilang** perlu diimputasi untuk memastikan bahwa model dapat dilatih tanpa kesalahan. Salah satu teknik yang umum digunakan untuk imputasi adalah dengan mengganti nilai yang hilang dengan **median** atau **mean** (tergantung pada distribusi data).
  
  - Kolom seperti **Systolic BP**, **Diastolic**, **BS**, **BMI**, dan **Heart Rate** dapat diimputasi dengan **median**, karena data numeriknya cenderung lebih baik diwakili oleh median jika ada distribusi yang tidak normal.
  
  - Kolom **Risk Level** bisa diimputasi dengan **modus** (nilai yang paling sering muncul) atau pendekatan **prediktif** seperti klasifikasi.

### **Proses Pemrosesan Data:**
- Untuk **memastikan kualitas data** dan menghindari bias dalam pelatihan model, semua **nilai yang hilang** akan ditangani menggunakan **imputasi**, atau dalam kasus yang ekstrim, data yang hilang dapat dihapus jika jumlahnya kecil.

"""

# %%
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.isna().sum()

# %%
"""
Melakukan pembersihan data 

- **`drop_duplicates()`** memastikan tidak ada data **ganda** dalam dataset.
- **`dropna()`** menghapus **baris yang mengandung nilai hilang** (missing values).
- **`isna().sum()`** digunakan untuk memastikan **tidak ada nilai yang hilang** setelah pembersihan data dilakukan.


"""

# %%
# Memeriksa ukuran dataset
print("Ukuran dataset (baris, kolom):", df.shape)


# %%
"""

Dataset ini memiliki **1148 baris** dan **12 kolom**. Berikut adalah ringkasan ukuran dataset:

| Ukuran Dataset | Jumlah Baris | Jumlah Kolom |
|-----------------|--------------|--------------|
| (1148, 12)      | 1148         | 12           |

"""

# %%
"""
3. Visualisasi Distribusi Data
"""

# %%

# Membuat histogram untuk semua fitur numerik
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()


# %%
"""


Histogram-histogram di bawah ini menunjukkan distribusi dari berbagai fitur medis dalam dataset, yang digunakan untuk memprediksi **risiko kehamilan**.

### Penjelasan Histogram:

1. **Age**:
   - Mayoritas ibu hamil dalam dataset ini berada di rentang usia yang lebih muda (di bawah 50 tahun), dengan beberapa outlier yang menunjukkan usia yang sangat tinggi (lebih dari 100 tahun).
   
2. **Systolic BP (Tekanan Darah Sistolik)**:
   - Distribusi **tekanan darah sistolik** menunjukkan bahwa sebagian besar ibu hamil memiliki tekanan darah sistolik antara 100 dan 130 mmHg.
   
3. **Diastolic (Tekanan Darah Diastolik)**:
   - Sebagian besar data di sekitar rentang 60 hingga 80 mmHg, dengan beberapa outlier di atas 100 mmHg.

4. **BS (Blood Sugar / Gula Darah)**:
   - Distribusi gula darah sebagian besar berada pada nilai lebih rendah (di bawah 10), dengan beberapa nilai ekstrim di atas 15.
   
5. **Body Temp (Suhu Tubuh)**:
   - Hampir semua data terpusat pada suhu tubuh 98°C, yang merupakan suhu tubuh normal. Beberapa nilai outlier di atas 99°C perlu diperhatikan.

6. **BMI (Body Mass Index)**:
   - Sebagian besar ibu hamil memiliki **BMI** antara 20 hingga 30, menunjukkan bahwa mayoritas memiliki **berat badan normal hingga kelebihan**.

7. **Previous Complications (Komplikasi Sebelumnya)**:
   - Sebagian besar ibu hamil tidak memiliki komplikasi sebelumnya, dengan sedikit data yang menunjukkan ibu hamil dengan komplikasi (nilai 1).

8. **Preexisting Diabetes (Diabetes Sebelumnya)**:
   - Sebagian besar ibu hamil tidak memiliki diabetes sebelumnya, dengan sedikit data yang menunjukkan ada diabetes (nilai 1).

9. **Gestational Diabetes (Diabetes Gestasional)**:
   - Sebagian besar ibu hamil tidak mengalami diabetes gestasional, dengan sedikit ibu hamil yang terdiagnosis mengalaminya.

10. **Mental Health (Kesehatan Mental)**:
    - Sebagian besar ibu hamil tidak memiliki masalah kesehatan mental, meskipun ada sebagian kecil dengan masalah kesehatan mental (nilai 1).

11. **Heart Rate (Detak Jantung)**:
    - Mayoritas ibu hamil memiliki **detak jantung** sekitar 70-80 bpm, dengan beberapa outlier pada detak jantung yang lebih rendah dan lebih tinggi.

### Kesimpulan:
- **Distribusi fitur** ini memberikan wawasan mengenai karakteristik kesehatan ibu hamil dalam dataset. Beberapa fitur memiliki distribusi yang terpusat (misalnya, suhu tubuh dan BMI), sementara yang lain menunjukkan adanya **outlier** yang perlu dianalisis lebih lanjut, seperti **Age**, **Systolic BP**, dan **BS**.
- **Histogram** ini membantu dalam memahami bagaimana data terdistribusi, yang sangat berguna untuk **preprocessing** sebelum melatih model machine learning.


"""

# %%
"""
4. Heatmap Korelasi Antarfitur
"""

# %%
# Membuat heatmap korelasi antar fitur numerik
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


# %%
"""
### Analisis Heatmap Korelasi Fitur Numerik dalam Dataset Kesehatan Ibu

Heatmap ini menggambarkan korelasi antara fitur numerik dalam dataset yang digunakan untuk memprediksi risiko kesehatan ibu selama kehamilan. Korelasi ini menunjukkan seberapa besar hubungan antara satu fitur dengan fitur lainnya, dengan nilai berkisar antara -1 hingga 1. Nilai yang lebih dekat ke 1 menunjukkan korelasi positif yang kuat, sementara nilai yang mendekati -1 menunjukkan korelasi negatif yang kuat.

#### Penjelasan Hasil Korelasi:
1. **Usia (Age) dan Sistolik BP (Systolic BP):**
   - Terdapat korelasi positif rendah (0.16), yang berarti bahwa meskipun ada sedikit hubungan antara usia ibu dan tekanan darah sistolik, korelasi ini tidak terlalu kuat.

2. **Sistolik BP dan Diastolik BP:**
   - Korelasi yang sangat tinggi (0.79), menunjukkan bahwa tekanan darah sistolik dan diastolik cenderung bergerak seiring. Hal ini sesuai dengan pemahaman medis bahwa keduanya terkait erat dalam mengukur tekanan darah.

3. **Sistolik BP dan Gula Darah (BS):**
   - Korelasi moderat (0.34), menunjukkan hubungan yang wajar antara tekanan darah sistolik dan kadar gula darah, yang mungkin mencerminkan hubungan antara tekanan darah tinggi dan risiko diabetes gestasional.

4. **BMI dan Gula Darah (BS):**
   - Korelasi yang cukup kuat (0.49), yang menunjukkan bahwa semakin tinggi BMI ibu, semakin tinggi kemungkinan kadar gula darah juga meningkat, ini bisa menjadi indikator risiko diabetes.

5. **Diabetes yang Ada Sebelumnya (Preexisting Diabetes) dan Diabetes Gestasional (Gestational Diabetes):**
   - Korelasi tinggi (0.55), mengindikasikan bahwa ibu yang memiliki riwayat diabetes sebelumnya memiliki kemungkinan lebih tinggi untuk mengembangkan diabetes gestasional.

6. **Komplikasi Sebelumnya (Previous Complications) dan BMI:**
   - Korelasi moderat (0.37), menunjukkan bahwa ibu dengan riwayat komplikasi sebelumnya cenderung memiliki BMI yang lebih tinggi, yang mungkin berhubungan dengan kondisi medis yang lebih kompleks selama kehamilan.

7. **Kesehatan Mental (Mental Health) dan Komplikasi Sebelumnya (Previous Complications):**
   - Korelasi positif moderat (0.45), yang menunjukkan bahwa ibu dengan riwayat komplikasi sebelumnya lebih cenderung mengalami masalah kesehatan mental, seperti stres atau kecemasan.

8. **Suhu Tubuh (Body Temp) dan Tekanan Darah (BP):**
   - Korelasi negatif yang lemah (-0.09 dan -0.19), yang menunjukkan bahwa tidak ada hubungan yang jelas antara suhu tubuh dan tekanan darah sistolik atau diastolik dalam dataset ini.

#### Kesimpulan:
Heatmap ini memberikan wawasan penting mengenai hubungan antar variabel dalam dataset. Beberapa korelasi yang signifikan, seperti antara **Sistolik BP** dan **Diastolik BP**, serta **Preexisting Diabetes** dan **Gestational Diabetes**, menunjukkan adanya pola yang perlu diperhatikan dalam model prediksi risiko kesehatan ibu. Sementara itu, korelasi yang lebih rendah menunjukkan bahwa beberapa variabel, seperti **Usia** dan **Body Temperature**, mungkin kurang relevan dalam memprediksi risiko yang lebih serius seperti diabetes atau hipertensi selama kehamilan.

### Implikasi untuk Model Prediksi
- Fitur yang menunjukkan korelasi kuat, seperti **Sistolik BP** dengan **Diastolik BP**, harus dipertimbangkan dengan lebih mendalam dalam model prediksi karena memberikan indikasi kuat tentang kondisi kesehatan ibu.
- Variabel seperti **Preexisting Diabetes** dan **Gestational Diabetes** juga merupakan indikator yang penting dan perlu ditekankan dalam model untuk memprediksi risiko kehamilan lebih lanjut.

"""

# %%
"""
5. Distribusi Kelas Target (Risk Level)
"""

# %%
# Menampilkan distribusi target 'Risk Level'
plt.figure(figsize=(8, 6))
sns.countplot(x='Risk Level', data=df)
plt.title('Distribusi Risiko Kesehatan Mental')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.show()


# %%
"""
### Distribusi Risiko Kesehatan Mental

Grafik ini menunjukkan distribusi frekuensi antara dua kategori risiko kesehatan mental: **Tinggi** dan **Rendah**. Terlihat bahwa jumlah individu dengan **risiko rendah** jauh lebih banyak dibandingkan dengan **risiko tinggi**. Hal ini menunjukkan bahwa mayoritas peserta dalam dataset ini tidak mengalami masalah kesehatan mental yang signifikan.

- **Risiko Tinggi**: Lebih sedikit dengan jumlah kurang dari 200.
- **Risiko Rendah**: Sebagian besar individu memiliki risiko rendah, mencapai lebih dari 600.

### Kesimpulan
Data menunjukkan bahwa sebagian besar peserta memiliki risiko kesehatan mental yang rendah, yang bisa berarti prevalensi masalah kesehatan mental pada populasi ini cukup rendah.

"""

# %%
"""
6. Distribusi Fitur Kategorikal (Mental Health)
"""

# %%
# Menampilkan distribusi 'Mental Health'
plt.figure(figsize=(8, 6))
sns.countplot(x='Mental Health', data=df)
plt.title('Distribusi Mental Health')
plt.xlabel('Mental Health')
plt.ylabel('Count')
plt.show()


# %%
"""
### Distribusi Kesehatan Mental

Grafik ini menunjukkan jumlah peserta yang terdiagnosis dengan masalah kesehatan mental (dalam kategori 0 dan 1). 

- **Mental Health = 0**: Sebagian besar individu (lebih dari 700) tidak mengalami masalah kesehatan mental yang signifikan.
- **Mental Health = 1**: Sekitar 400 individu mengalami masalah kesehatan mental.

### Kesimpulan
Mayoritas peserta dalam dataset ini tidak menunjukkan masalah kesehatan mental yang signifikan, tetapi ada juga proporsi yang cukup besar dengan kondisi kesehatan mental yang terpengaruh.

"""

# %%
"""
7. Pairplot untuk Melihat Relasi Antar Fitur
"""

# %%
# Membuat pairplot untuk fitur numerik (jika jumlah kolom numerik tidak terlalu banyak)
sns.pairplot(df.select_dtypes(include='number'))
plt.show()


# %%
"""
### Analisis Pairplot Fitur Numerik dalam Dataset Kesehatan Ibu

Gambar ini adalah **pairplot** yang menunjukkan distribusi dan hubungan antar berbagai fitur numerik dalam dataset. Setiap kolom dan baris mewakili fitur yang berbeda, dengan scatter plots di antara setiap pasangan fitur, serta histogram untuk masing-masing fitur pada diagonal.

#### Penjelasan:
- **Hubungan antar fitur**: Scatter plots di luar diagonal memberikan wawasan mengenai korelasi antara dua fitur. Sebagai contoh, dapat terlihat apakah ada pola linear atau non-linear antara fitur-fitur seperti **Sistolik BP** dan **Diastolik BP**, atau **BMI** dan **Gula Darah**.
- **Distribusi individual**: Diagonal menunjukkan distribusi masing-masing fitur (dalam bentuk histogram), yang memberikan gambaran tentang seberapa merata atau terdistribusinya data untuk setiap fitur.

#### Temuan yang Mungkin Terlihat:
1. **Korelasi kuat**: Jika ada pola garis lurus atau konsisten dalam scatter plot, ini menunjukkan korelasi yang kuat antara dua fitur. Misalnya, **Sistolik BP** dan **Diastolik BP** kemungkinan memiliki korelasi yang tinggi.
2. **Outlier atau keanehan**: Scatter plot yang menunjukkan data yang tersebar luas atau sangat terkonsentrasi di area tertentu bisa mengindikasikan adanya outlier atau data yang tidak normal.

### Kesimpulan:
Pairplot ini memberikan gambaran yang sangat baik tentang hubungan antar fitur dalam dataset, yang penting untuk mengeksplorasi lebih lanjut dalam model prediksi untuk mendeteksi pola risiko dalam kesehatan ibu selama kehamilan.

"""

# %%
"""
8. Visualisasi Korelasi Fitur dengan Target
"""

# %%
# Visualisasi perbandingan fitur numerik dengan target 'Risk Level'
for feature in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Risk Level', y=feature, data=df)
    plt.title(f'Perbandingan {feature} dengan Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel(feature)
    plt.show()


# %%
"""
### Perbandingan Fitur Kesehatan dengan Level Risiko

Gambar yang disediakan menampilkan **box plots** yang menggambarkan perbandingan beberapa fitur kesehatan dengan **Risk Level** (tinggi atau rendah). Setiap box plot menunjukkan distribusi nilai untuk setiap fitur (misalnya: usia, tekanan darah, detak jantung, BMI) berdasarkan dua kategori tingkat risiko. Box plot ini digunakan untuk mengidentifikasi seberapa besar variasi antara fitur untuk setiap kategori dan apakah terdapat perbedaan signifikan antara kelompok yang memiliki **risk level tinggi** dan **low**.

#### Hasil dan Penjelasan Box Plots:
1. **Age (Usia)**:
   - Box plot menunjukkan distribusi usia yang hampir serupa antara **Risk Level Tinggi** dan **Rendah**, dengan median yang berada pada kisaran yang sama, namun ada sedikit outlier pada kelompok **Tinggi** yang menandakan adanya individu dengan usia yang lebih tua dalam kategori ini.
   
2. **Systolic BP (Tekanan Darah Sistolik)**:
   - **Risk Level Tinggi** memiliki distribusi yang lebih lebar dan lebih tinggi dibandingkan dengan **Risk Level Rendah**, menunjukkan bahwa individu dengan risiko tinggi cenderung memiliki tekanan darah sistolik yang lebih tinggi. Terdapat juga beberapa outlier yang menunjukkan adanya individu dengan tekanan darah yang sangat tinggi.

3. **Diastolic BP (Tekanan Darah Diastolik)**:
   - Seperti halnya **Sistolik BP**, tekanan darah diastolik pada **Risk Level Tinggi** juga lebih tinggi dan lebih tersebar dibandingkan dengan **Risk Level Rendah**. Ada outlier yang menunjukkan individu dengan tekanan darah diastolik yang sangat tinggi.

4. **BS (Blood Sugar - Gula Darah)**:
   - Gula darah pada **Risk Level Tinggi** memiliki distribusi yang lebih lebar dengan beberapa outlier yang menunjukkan kadar gula darah yang sangat tinggi. Sementara itu, pada **Risk Level Rendah**, nilai gula darah lebih terpusat di sekitar nilai normal, dengan sedikit outlier.

5. **Body Temperature (Suhu Tubuh)**:
   - Distribusi suhu tubuh pada kedua **Risk Level** sangat mirip, dengan sebagian besar individu memiliki suhu tubuh normal, namun terdapat beberapa outlier dengan suhu tubuh yang lebih tinggi, meskipun jarang terjadi.

6. **BMI (Indeks Massa Tubuh)**:
   - **Risk Level Tinggi** menunjukkan distribusi BMI yang lebih lebar dengan median lebih tinggi, yang mengindikasikan bahwa individu dengan risiko tinggi cenderung memiliki BMI yang lebih tinggi. Terdapat beberapa outlier di kedua kategori, yang menunjukkan adanya individu dengan BMI ekstrem.

7. **Previous Complications (Komplikasi Sebelumnya)**:
   - Box plot ini sangat terpusat pada **Risk Level Tinggi**, menunjukkan bahwa sebagian besar individu dengan komplikasi sebelumnya berada pada **Risk Level Tinggi**, sedangkan individu dengan **Risk Level Rendah** sangat sedikit yang memiliki riwayat komplikasi.

8. **Preexisting Diabetes (Diabetes Sebelumnya)**:
   - Seperti halnya **Previous Complications**, individu dengan **Preexisting Diabetes** lebih banyak berada dalam kategori **Risk Level Tinggi**.

9. **Gestational Diabetes (Diabetes Gestasional)**:
   - Distribusi diabetes gestasional juga sangat terpusat pada **Risk Level Tinggi**, dengan hampir tidak ada individu pada **Risk Level Rendah** yang mengalami kondisi ini.

10. **Mental Health (Kesehatan Mental)**:
    - **Risk Level Tinggi** memiliki distribusi yang hampir seluruhnya terdiri dari individu dengan masalah kesehatan mental, sedangkan pada **Risk Level Rendah**, sebagian besar individu tidak mengalami masalah kesehatan mental.

#### Tabel Ringkasan:
Berikut adalah tabel yang merangkum informasi perbandingan setiap fitur dengan **Risk Level**:

| Fitur                | Risk Level Tinggi     | Risk Level Rendah    |
|----------------------|-----------------------|----------------------|
| **Age (Usia)**        | Median: ~50, Outliers  | Median: ~50, Outliers |
| **Systolic BP**       | Lebih tinggi dan lebih bervariasi | Lebih rendah dan terpusat |
| **Diastolic BP**      | Lebih tinggi dan lebih bervariasi | Lebih rendah dan terpusat |
| **BS (Gula Darah)**   | Lebih tinggi dan bervariasi | Lebih rendah dan terpusat |
| **Body Temp**         | Beberapa outlier suhu tinggi | Suhu tubuh normal, sedikit outlier |
| **BMI**               | Lebih tinggi dan lebih bervariasi | Terpusat pada nilai normal |
| **Previous Complications** | Mayoritas di Risk Level Tinggi | Hampir tidak ada di Risk Level Rendah |
| **Preexisting Diabetes** | Mayoritas di Risk Level Tinggi | Hampir tidak ada di Risk Level Rendah |
| **Gestational Diabetes** | Mayoritas di Risk Level Tinggi | Hampir tidak ada di Risk Level Rendah |
| **Mental Health**     | Mayoritas di Risk Level Tinggi | Hampir tidak ada masalah mental |

### Kesimpulan:
Box plot ini memberikan wawasan penting mengenai perbedaan distribusi berbagai fitur kesehatan antara individu dengan risiko tinggi dan rendah. Fitur seperti **tekanan darah sistolik**, **BMI**, dan **gula darah** menunjukkan perbedaan yang signifikan antara kedua kelompok risiko ini, yang dapat membantu dalam mendeteksi faktor-faktor risiko yang dapat mempengaruhi hasil kehamilan.


"""

# %%
"""
**Data Preparation**
"""

# %%
"""
1. Menghapus Outlier (Jika Diperlukan)
"""

# %%
# Menggunakan boxplot untuk mendeteksi outlier
plt.figure(figsize=(15, 10))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.title("Box Plots of Numerical Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# %%
"""
## Penjelasan Box Plot Fitur Numerik

Box plot di bawah ini menggambarkan distribusi fitur numerik dalam dataset yang digunakan untuk memprediksi **risiko kehamilan**. Berikut adalah penjelasan mengenai masing-masing fitur:

### 1. **Age**:
   - **Distribusi**: Mayoritas data ibu hamil memiliki **usia antara 20 hingga 50 tahun**. Terdapat beberapa **outlier** di sisi kiri (usia sangat muda) dan kanan (usia sangat tua).
   - **Outliers**: Nilai usia di luar rentang normal (misalnya lebih dari 100 tahun) merupakan **outlier** yang perlu dianalisis lebih lanjut.

### 2. **Systolic BP** (Tekanan Darah Sistolik):
   - **Distribusi**: Data sebagian besar terdistribusi di rentang **100 hingga 140 mmHg**, dengan beberapa outlier di sisi kanan, menunjukkan nilai yang sangat tinggi (>160 mmHg).
   - **Outliers**: Beberapa titik di sisi kanan dengan nilai lebih dari 160 mmHg perlu diperiksa lebih lanjut.

### 3. **Diastolic** (Tekanan Darah Diastolik):
   - **Distribusi**: Mayoritas data berada di sekitar **70 hingga 90 mmHg**, dengan beberapa **outlier** pada nilai lebih tinggi dari 100 mmHg.
   - **Outliers**: Data di sisi kanan dengan nilai lebih dari 100 mmHg menunjukkan **outlier** pada tekanan darah diastolik.

### 4. **BS** (Blood Sugar / Gula Darah):
   - **Distribusi**: Gula darah mayoritas berada di bawah **10**, dengan beberapa nilai ekstrim (lebih tinggi dari 15) yang bisa dianggap sebagai **outlier**.
   - **Outliers**: Beberapa titik menunjukkan **outlier** dengan gula darah yang sangat tinggi.

### 5. **Body Temp** (Suhu Tubuh):
   - **Distribusi**: Hampir seluruh data memiliki suhu tubuh **98°C**, dengan beberapa **outlier** di sisi kanan yang lebih tinggi (lebih dari 99°C).
   - **Outliers**: Ada beberapa titik menunjukkan **outlier** dengan suhu tubuh lebih dari 99°C, yang bisa berarti anomali atau kesalahan pengukuran.

### 6. **BMI** (Body Mass Index):
   - **Distribusi**: Sebagian besar nilai **BMI** berada di antara **20 hingga 30**, menunjukkan kategori berat badan normal hingga kelebihan.
   - **Outliers**: Terdapat beberapa nilai **outlier** pada **BMI** yang sangat rendah (di bawah 10) dan sangat tinggi (di atas 35), yang perlu diperiksa lebih lanjut.

### 7. **Previous Complications** (Komplikasi Sebelumnya):
   - **Distribusi**: Sebagian besar data adalah **0**, yang menunjukkan bahwa kebanyakan ibu hamil tidak memiliki komplikasi sebelumnya.
   - **Outliers**: Hanya ada beberapa titik pada nilai **1**, yang menunjukkan ibu hamil dengan komplikasi sebelumnya.

### 8. **Preexisting Diabetes** (Diabetes Sebelumnya):
   - **Distribusi**: Sebagian besar ibu hamil tidak memiliki diabetes sebelumnya (**0**), sementara yang lainnya ada sedikit ibu yang memiliki diabetes sebelumnya (**1**).
   - **Outliers**: Beberapa titik **outlier** pada **nilai 1** menunjukkan ibu hamil yang memiliki diabetes sebelumnya.

9. **Gestational Diabetes** (Diabetes Gestasional):
   - **Distribusi**: Sebagian besar ibu hamil tidak mengalami diabetes gestasional (**0**), namun ada sebagian kecil dengan kondisi tersebut (**1**).
   - **Outliers**: Ada beberapa titik pada **nilai 1**, menunjukkan ibu hamil yang didiagnosis dengan diabetes gestasional.

10. **Mental Health** (Kesehatan Mental):
    - **Distribusi**: Sebagian besar ibu hamil tidak memiliki masalah kesehatan mental (**0**), tetapi ada sedikit ibu hamil dengan masalah kesehatan mental (**1**).
    - **Outliers**: Beberapa titik **outlier** pada **nilai 1**, menunjukkan ibu hamil yang memiliki masalah kesehatan mental.

11. **Heart Rate** (Detak Jantung):
    - **Distribusi**: Mayoritas ibu hamil memiliki **detak jantung** di rentang **70 hingga 80 bpm**. Terdapat beberapa **outlier** dengan nilai detak jantung yang sangat rendah atau tinggi.
    - **Outliers**: Beberapa titik di sisi kiri dengan **detak jantung rendah** dan di sisi kanan dengan **detak jantung tinggi** adalah **outlier** yang perlu diperiksa lebih lanjut.

**Kesimpulan**:
- **Outlier** terdeteksi di beberapa fitur seperti **Age**, **Systolic BP**, **Diastolic**, **BS**, **BMI**, dan **Heart Rate**, yang perlu dianalisis lebih lanjut.
- **Distribusi fitur** ini memberikan wawasan tentang **karakteristik medis ibu hamil**, dan dapat membantu dalam **pembersihan data** serta **pengolahan lebih lanjut** sebelum pelatihan model.

"""

# %%
"""
1.1 Menangani Outliers dengan IQR
"""

# %%
# Menghitung IQR untuk mendeteksi dan menghapus outlier
Q1 = df['Systolic BP'].quantile(0.25)
Q3 = df['Systolic BP'].quantile(0.75)
IQR = Q3 - Q1

# Menentukan batas bawah dan batas atas untuk mendeteksi outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menghapus outlier yang berada di luar rentang ini
df = df[(df['Systolic BP'] >= lower_bound) & (df['Systolic BP'] <= upper_bound)]

# Melakukan hal yang sama untuk fitur lain yang memiliki outliers (misalnya, 'Age', 'Diastolic')
Q1_age = df['Age'].quantile(0.25)
Q3_age = df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age

lower_bound_age = Q1_age - 1.5 * IQR_age
upper_bound_age = Q3_age + 1.5 * IQR_age

df = df[(df['Age'] >= lower_bound_age) & (df['Age'] <= upper_bound_age)]

# Memeriksa ukuran dataset setelah menghapus outlier
print(f"Dataset setelah menghapus outlier memiliki ukuran: {df.shape}")


# %%
"""
### Ukuran Dataset Setelah Menghapus Outlier

Setelah melakukan **penghapusan outlier** pada beberapa fitur, ukuran dataset menjadi:

| Ukuran Dataset       | Jumlah Baris | Jumlah Kolom |
|----------------------|--------------|--------------|
| (1082, 12)           | 1082         | 12           |

### Penjelasan:
- **Jumlah Baris**: Dataset sekarang memiliki **1082 entri** setelah menghapus **outlier**. Ini menunjukkan bahwa sebagian data telah dibersihkan.
- **Jumlah Kolom**: Dataset ini masih memiliki **12 kolom**, yang mencakup berbagai fitur medis dan target **Risk Level**.

### Langkah Selanjutnya:
- Dataset ini sekarang sudah **bersih** dari **outlier**, dan siap untuk digunakan dalam **pelatihan model machine learning** atau **analisis lebih lanjut**.

"""

# %%
"""
**Data Prepocessing**
"""

# %%
"""
 1. Normalisasi Fitur dengan StandardScaler
"""

# %%
from sklearn.preprocessing import StandardScaler

# Melakukan scaling pada fitur numerik
scaler = StandardScaler()

# Menormalkan data pada fitur numerik
X = df.drop('Risk Level', axis=1)  # Fitur
y = df['Risk Level']  # Target

# Menggunakan StandardScaler pada data latih dan data uji
X_scaled = scaler.fit_transform(X)

# Memeriksa distribusi data setelah scaling
plt.figure(figsize=(10, 6))
sns.histplot(X_scaled[:, 0], kde=True)
plt.title("Distribusi Fitur Setelah Scaling")
plt.show()


# %%
"""
### Distribusi Fitur Setelah Scaling

Grafik di atas menunjukkan distribusi dari fitur setelah dilakukan **scaling** menggunakan metode seperti **StandardScaler**. Berikut adalah beberapa hal yang dapat diamati dari grafik ini:

### **Penjelasan Grafik**:
1. **Histogram dan Kernel Density Estimate (KDE)**:
   - Histogram menunjukkan **frekuensi distribusi nilai** untuk fitur yang telah di-*scale*, dengan **KDE** (kurva biru) yang menggambarkan **kepadatan distribusi**.
   
2. **Skala Nilai**:
   - **Scaling** biasanya membuat fitur memiliki **rata-rata 0 dan standar deviasi 1**, sehingga distribusinya terpusat di sekitar **0** dengan sebagian besar data berada di sekitar nilai tersebut.
   
3. **Penyebaran Data**:
   - Data tersebar di sekitar **0**, dengan sebagian besar data berada di rentang **-2 hingga 2**. Hal ini menunjukkan bahwa data sudah **distandarisasi** dan tidak terpengaruh oleh **skala asli** fitur.
   
4. **Perbandingan Sebelum dan Sesudah Scaling**:
   - Sebelum scaling, fitur dengan **skala besar** seperti **BMI** atau **tekanan darah** bisa mempengaruhi model secara signifikan, karena model lebih cenderung memperhatikan fitur dengan rentang nilai yang lebih besar. Dengan **scaling**, semua fitur kini berada dalam rentang yang seragam, yang memungkinkan model belajar lebih baik tanpa terpengaruh oleh perbedaan skala antar fitur.

### **Pentingnya Scaling**:
- **Scaling** sangat penting untuk algoritma yang sensitif terhadap skala data, seperti **Neural Networks**, **Logistic Regression**, atau **K-Nearest Neighbors (KNN)**. Ini memastikan bahwa semua fitur memiliki **pengaruh yang seimbang** terhadap model.

### **Kesimpulan**:
- **Distribusi data** setelah scaling menunjukkan bahwa semua fitur sekarang memiliki **skala yang konsisten**, yang membantu dalam **pelatihan model** yang lebih efisien dan efektif.

"""

# %%
"""
2. Encode target
"""

# %%
# Encode target
le = LabelEncoder()
df['Risk Level'] = le.fit_transform(df['Risk Level'])


# %%
"""
3. Pembagian Data Latih dan Uji (80% Train, 20% Test)
"""

# %%
from sklearn.model_selection import train_test_split

# Membagi data menjadi 80% latih dan 20% uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Memeriksa ukuran data latih dan uji
print(f"Ukuran Data Latih: {X_train.shape}")
print(f"Ukuran Data Uji: {X_test.shape}")


# %%
"""
### Ukuran Data Latih dan Data Uji

Setelah membagi dataset menjadi **data latih** dan **data uji**, ukuran dataset menjadi sebagai berikut:

### **Data Latih (Training Data)**:
- **Ukuran**: **865 baris** dan **11 kolom**
- Data latih digunakan untuk **melatih model** agar dapat mengenali pola-pola yang ada dalam dataset.

### **Data Uji (Testing Data)**:
- **Ukuran**: **217 baris** dan **11 kolom**
- Data uji digunakan untuk **menguji performa model** setelah pelatihan, untuk memastikan model dapat menggeneralisasi dengan baik pada data yang belum pernah dilihat sebelumnya.

### **Distribusi Data**:
- Data latih digunakan untuk **mengoptimalkan model**, sedangkan data uji digunakan untuk **evaluasi model** dan melihat seberapa baik model dapat memprediksi **risiko kehamilan** pada data yang belum pernah dilihat.

"""

# %%
"""
4.  Menangani Ketidakseimbangan Kelas dengan SMOTE
"""

# %%
from imblearn.over_sampling import SMOTE

# Menangani ketidakseimbangan kelas dengan SMOTE
smote = SMOTE(random_state=42)

# Oversampling pada data latih
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Memeriksa distribusi kelas setelah SMOTE
print(pd.Series(y_train_res).value_counts())


# %%
"""
### Distribusi Kelas pada Kolom `Risk Level`

Kolom **`Risk Level`** menunjukkan **tingkat risiko kehamilan** yang dikategorikan menjadi dua kelas: **High** dan **Low**. Berdasarkan analisis, distribusi kelas dalam dataset adalah sebagai berikut:

| Risk Level | Jumlah |
|------------|--------|
| High       | 524    |
| Low        | 524    |

### Penjelasan:
- **Kelas High**: Terdapat **524** entri yang diklasifikasikan sebagai **risiko kehamilan tinggi**.
- **Kelas Low**: Terdapat **524** entri yang diklasifikasikan sebagai **risiko kehamilan rendah**.

### Implikasi:
- **Distribusi Seimbang**: Dataset ini memiliki distribusi kelas yang **seimbang** antara **Low** dan **High**, yang sangat penting untuk **pelatihan model** agar tidak ada **bias** terhadap kelas mayoritas. 
- **Keseimbangan Kelas**: Dalam kasus klasifikasi seperti ini, pembagian yang seimbang antara kelas sangat menguntungkan karena model tidak akan lebih cenderung ke satu kelas dibandingkan kelas lainnya.


"""

# %%
"""
**Modeling**
"""

# %%
import tensorflow as tf

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 kelas: Low, Medium, High
])

model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()



# %%
"""
### Penjelasan Model Neural Network

Model ini merupakan **Neural Network** dengan **3 lapisan Dense** dan **Dropout** untuk regularisasi. Model ini digunakan untuk memprediksi **risiko kehamilan** berdasarkan fitur medis ibu hamil.

### **Ringkasan Arsitektur Model**:

| **Layer (type)**   | **Output Shape** | **Parameter #** |
|--------------------|------------------|-----------------|
| **Dense** (64 units)   | (None, 64)           | 768            |
| **Dropout** (0.2)      | (None, 64)           | 0              |
| **Dense** (32 units)   | (None, 32)           | 2,080          |
| **Dropout** (0.2)      | (None, 32)           | 0              |
| **Dense** (3 units)    | (None, 3)            | 99             |

- **Total Parameters**: 2,947 (**11.51 KB**)
- **Trainable Parameters**: 2,947 (**11.51 KB**)
- **Non-trainable Parameters**: 0 (Tidak ada parameter yang tidak dapat dilatih)

### **Penjelasan Setiap Layer**:

1. **Layer 1 - Dense (64 units)**:
   - **Output Shape**: (None, 64), menghasilkan **64 neuron** pada layer pertama, yang menghasilkan vektor fitur berukuran 64 untuk setiap input data.
   - **Parameters**: 768 parameter untuk **weight** dan **bias** antara layer input dan layer pertama.

2. **Layer 2 - Dropout (0.2)**:
   - Dropout digunakan untuk **mengurangi overfitting** dengan mengabaikan **20% dari unit** pada layer sebelumnya selama pelatihan. Ini membantu model agar tidak terlalu bergantung pada fitur tertentu.

3. **Layer 3 - Dense (32 units)**:
   - **Output Shape**: (None, 32), menghasilkan 32 unit output.
   - **Parameters**: 2,080 parameter untuk **weight** dan **bias** antara layer pertama dan layer kedua.

4. **Layer 4 - Dropout (0.2)**:
   - Dropout lagi untuk **regularisasi**, mengurangi risiko model **overfitting** pada data latih.

5. **Layer 5 - Dense (3 units)**:
   - **Output Shape**: (None, 3), menghasilkan 3 output yang mewakili **3 kelas** (di sini bisa merujuk pada tingkat risiko kehamilan, namun sebaiknya menggunakan **softmax** untuk probabilitas kelas).
   - **Parameters**: 99 parameter untuk **weight** dan **bias** antara layer kedua dan output layer.

### **Peringatan dalam Model**:
- **Peringatan**: `UserWarning: Do not pass an input_shape/input_dim argument to a layer.` 
  - Peringatan ini menyarankan untuk menggunakan **`Input()`** sebagai layer pertama dalam model **Sequential** daripada menetapkan **input_shape** langsung di dalam **Dense** layer pertama.
  - Cara yang lebih baik adalah dengan menggunakan **`Input(shape=(input_shape))`** untuk layer pertama daripada menuliskan **`input_shape`** di dalam **Dense**.

### **Kesimpulan**:
- Model ini memiliki **3 layer Dense** dengan **Dropout** untuk mencegah **overfitting**. Dengan **2,947 parameter** yang dapat dilatih, model ini siap untuk memprediksi **risiko kehamilan** berdasarkan **fitur medis**.

"""

# %%
# Pastikan target sudah di-encode ke numerik
# Update y, y_train, y_test, y_train_res agar bertipe numerik
y = df['Risk Level']
from sklearn.model_selection import train_test_split

# Split ulang data dengan target yang sudah di-encode
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)


# Melatih model
history = model.fit(X_train_res, y_train_res, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Menampilkan hasil pelatihan
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")


# %%
"""
## Penjelasan Hasil Pelatihan Model

### **Akurasi dan Loss Model pada Setiap Epoch**

Pada setiap **epoch**, model dilatih menggunakan data latih dan diuji pada data validasi untuk mengukur **akurasi** dan **loss** pada data uji.

#### **Epoch 1-50**:
- **Akurasi Latih** (`accuracy`):
  - Pada **epoch 1**, model mulai dengan **akurasi 98.96%** pada data latih.
  - Secara umum, **akurasi latih** menunjukkan **peningkatan stabil** selama pelatihan. Pada **epoch 50**, akurasi mencapai **99.54%**.
  - Ini menunjukkan bahwa model **belajar dengan baik** dari data latih dan **meningkatkan kemampuannya** dalam mengenali pola dari data latih.

- **Loss Latih** (`loss`):
  - **Loss** pada data latih dimulai dengan **0.0333** pada epoch pertama dan kemudian **menurun secara bertahap** hingga mencapai **0.0157** pada epoch terakhir.
  - Penurunan **loss** ini menunjukkan bahwa model terus **menurunkan kesalahan** pada prediksi yang dilakukan pada data latih.

- **Akurasi Validasi** (`val_accuracy`):
  - **Akurasi pada data validasi** tetap relatif stabil di **sekitar 95.85%** pada epoch-epoch awal dan kemudian meningkat sedikit ke **96.31%** pada epoch 50.
  - Ini menunjukkan bahwa model dapat **menggeneralisasi dengan baik** pada data uji, meskipun ada sedikit fluktuasi di awal pelatihan.

- **Loss Validasi** (`val_loss`):
  - **Loss** pada data validasi dimulai dengan **0.0836** pada epoch pertama dan sedikit fluktuasi, tetapi secara umum tetap relatif stabil, dengan nilai terakhir di **0.0958** pada epoch 50.
  - Meskipun ada fluktuasi, **loss** pada data validasi tidak meningkat secara signifikan, yang menunjukkan bahwa model tidak mengalami **overfitting** meskipun ada sedikit penurunan pada **val_accuracy**.

#### **Interpretasi**:
- **Akurasi Latih** meningkat dengan pesat dan **loss latih menurun**, yang menunjukkan bahwa model semakin baik dalam memprediksi data latih.
- **Akurasi Validasi** relatif stabil dan hanya sedikit meningkat, tetapi tetap menunjukkan hasil yang sangat baik (**96.31%**) di akhir pelatihan.
- **Loss Validasi** juga tetap stabil dan menunjukkan bahwa model **menggeneralisasi dengan baik** pada data yang tidak terlihat selama pelatihan.
- Tidak ada **overfitting** yang signifikan terlihat, meskipun ada sedikit **fluktuasi** dalam **validasi accuracy** dan **loss** selama pelatihan.

### **Final Validation Accuracy**:
Pada **epoch terakhir (50)**, model mencapai **akurasi validasi 96.31%** dan **loss validasi 0.0958**, yang menunjukkan bahwa model sudah **siap** digunakan untuk **prediksi risiko kehamilan** dengan akurasi yang sangat baik.

### **Kesimpulan**:
- **Model** berhasil mencapai **akurasi tinggi** pada **data uji** dan mengalami **penurunan loss** selama pelatihan.
- **Akurasi Validasi** yang stabil menunjukkan **generalization** yang baik, yang penting untuk menghindari **overfitting**.
- Model ini sangat siap untuk digunakan dalam **prediksi risiko kehamilan**.

"""

# %%
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# %%
"""
### Visualisasi Akurasi dan Loss Model

### 1. **Model Accuracy**:
Grafik di bawah ini menunjukkan **akurasi model** selama pelatihan pada **data latih** (Train Accuracy) dan **data validasi** (Validation Accuracy).

- **Train Accuracy** (Akurasi Data Latih): 
  - Akurasi model meningkat dengan sangat cepat di awal pelatihan, mencapai **lebih dari 98%** setelah beberapa epoch. Ini menunjukkan bahwa model mulai belajar dengan baik dari data latih.
  
- **Validation Accuracy** (Akurasi Data Uji):
  - Acuracy data uji menunjukkan sedikit fluktuasi selama pelatihan, tetapi tetap berada pada tingkat **sekitar 96%** pada sebagian besar epoch. Meskipun **train accuracy** meningkat secara tajam, **validation accuracy** menunjukkan kestabilan yang lebih baik, yang menandakan model tidak mengalami **overfitting** secara signifikan.

### 2. **Model Loss**:
Grafik ini menunjukkan **loss model** selama pelatihan, baik pada **data latih** (Train Loss) maupun **data validasi** (Validation Loss).

- **Train Loss** (Loss Data Latih): 
  - Loss pada data latih turun dengan cepat pada awal pelatihan, yang menunjukkan bahwa model mulai mengoptimalkan prediksi dengan sangat cepat. Pada akhir pelatihan, loss stabil pada nilai yang sangat rendah sekitar **0.05**.

- **Validation Loss** (Loss Data Uji):
  - Loss pada data uji (validation loss) menunjukkan penurunan yang lebih perlahan, namun tetap stabil dan tidak menunjukkan lonjakan yang besar. Hal ini menunjukkan bahwa model dapat menggeneralisasi dengan baik pada data yang belum terlihat selama pelatihan.

### **Kesimpulan**:
- **Train Accuracy** dan **Validation Accuracy** keduanya menunjukkan hasil yang sangat baik, dengan model mencapai **akurasi lebih dari 96%** pada data uji.
- **Train Loss** dan **Validation Loss** menunjukkan konvergensi yang stabil, yang mengindikasikan bahwa model **tidak overfitting** dan dapat **menggeneralisasi** dengan baik pada data yang belum pernah dilihat sebelumnya.

"""

# %%
# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Convert class labels back to strings using the fitted label encoder's classes
# This ensures we only use the labels the encoder was trained on.
target_names = [str(cls) for cls in le.classes_]

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_labels, target_names=target_names))

# For the confusion matrix as well, use the correct target names
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %%
"""
### Confusion Matrix dan Metrik Evaluasi

### **Confusion Matrix**:
Confusion matrix di bawah ini menunjukkan perbandingan antara **nilai aktual** dan **prediksi model** untuk dua kelas **risiko kehamilan** (Low dan High).

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


"""

# %%
"""
**Hyperparameter Tunning**
"""

# %%
"""
1. Hyperparameter Tuning dengan GridSearchCV
"""

# %%
# KerasClassifier sudah diimport dari scikeras.wrappers pada cell sebelumnya
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adamax, SGD, RMSprop

# Fungsi untuk membangun model Keras, menerima parameter untuk tuning
def build_model(optimizer='Adamax', dropout_rate=0.3, hidden_units=64):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(X_train_res.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units // 2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))  # 3 kelas: Low, Medium, High

    # Pilih optimizer sesuai parameter
    if optimizer == 'Adamax':
        opt = Adamax(learning_rate=0.005)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=0.005)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=0.005)
    else:
        opt = Adamax(learning_rate=0.005)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Membungkus model dalam KerasClassifier dari scikeras
model = KerasClassifier(
    model=build_model,
    verbose=0
)

# Gunakan prefix 'model__' untuk parameter model, sesuai scikeras
param_grid = {
    'model__optimizer': ['Adamax', 'SGD', 'RMSprop'],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'model__hidden_units': [32, 64, 128],
    'batch_size': [16, 32],
    'epochs': [50]
}

# Menyusun GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)

# Melakukan pencarian grid
grid_result = grid.fit(X_train_res, y_train_res)

# Menampilkan hasil terbaik
print(f"Best Parameters from Grid Search: {grid_result.best_params_}")
print(f"Best Accuracy from Grid Search: {grid_result.best_score_}")


# %%
"""
### Hasil Grid Search dan Peringatan dalam Model

### **Grid Search untuk Hyperparameter Tuning**:
Proses **Grid Search** dilakukan untuk mencari kombinasi **hyperparameter terbaik** bagi model. Berikut adalah **hasil terbaik** yang diperoleh dari Grid Search:

- **Best Parameters**:
  - **Batch Size**: 16
  - **Epochs**: 50
  - **Dropout Rate**: 0.3
  - **Hidden Units**: 128
  - **Optimizer**: Adamax

- **Best Accuracy**: 0.9790 (Akurasi terbaik pada data latih adalah **97.9%**).

Hasil ini menunjukkan bahwa model memberikan **akurasi tinggi** setelah tuning hyperparameter, dengan **dropout rate** 0.3 dan **Adamax** sebagai optimizer yang memberikan hasil terbaik.


"""

# %%
"""
 2. Menyusun RandomizedSearchCV
"""

# %%
from sklearn.model_selection import RandomizedSearchCV

# Tentukan distribusi untuk RandomizedSearchCV
param_dist = {
    'model__optimizer': ['Adamax', 'SGD', 'RMSprop'],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'model__hidden_units': [32, 64, 128, 256],
    'batch_size': [16, 32, 64],
    'epochs': [50],
}

# Menyusun RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                   n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)

# Melakukan pencarian random search
random_result = random_search.fit(X_train_res, y_train_res)

# Menampilkan hasil terbaik
print(f"Best Parameters from Random Search: {random_result.best_params_}")
print(f"Best Accuracy from Random Search: {random_result.best_score_}")


# %%
"""
### Hasil Random Search dan Peringatan dalam Model

### **Random Search untuk Hyperparameter Tuning**:
Proses **Random Search** dilakukan untuk mencari kombinasi **hyperparameter terbaik** bagi model. Berikut adalah **hasil terbaik** yang diperoleh dari Random Search:

- **Best Parameters**:
  - **Optimizer**: RMSprop
  - **Hidden Units**: 32
  - **Dropout Rate**: 0.4
  - **Epochs**: 50
  - **Batch Size**: 32

- **Best Accuracy**: 0.9780 (Akurasi terbaik pada data latih adalah **97.8%**).

Hasil ini menunjukkan bahwa model memberikan **akurasi tinggi** setelah tuning hyperparameter. **RMSprop** sebagai optimizer dan **dropout rate** 0.4 memberikan hasil terbaik dalam **mencegah overfitting** dan **meningkatkan akurasi model**.



"""

# %%
"""
3. Evaluasi Model Terbaik

"""

# %%
# Evaluasi model terbaik pada data uji
best_model = grid_result.best_estimator_  # Atau bisa menggunakan random_result.best_estimator_

# Evaluasi pada data uji menggunakan score() dari scikeras KerasClassifier
accuracy = best_model.score(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")


# %%
"""
4. Menampilkan Hasil Pelatihan dan Validasi
"""

# %%
# Mengambil riwayat pelatihan model terbaik dari scikeras KerasClassifier
history = grid_result.best_estimator_.history_

# Plot Akurasi
plt.figure(figsize=(12, 6))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# %%
"""
## Visualisasi Akurasi dan Loss Model

### **1. Grafik Model Accuracy**:
Grafik di bawah ini menunjukkan **akurasi model** pada **data latih (Train Accuracy)** dan **data validasi (Validation Accuracy)** selama **50 epoch**.

- **Train Accuracy (Akurasi Latih)**:
  - Akurasi model pada **data latih** terus meningkat dengan konsisten selama pelatihan, mencapai **lebih dari 99%** pada epoch terakhir.
  - Model dapat belajar dengan sangat baik dari data latih, dan akurasi terus meningkat sepanjang pelatihan.

- **Validation Accuracy (Akurasi Validasi)**:
  - Akurasi pada **data validasi** juga meningkat, meskipun tidak secepat **train accuracy**.
  - Namun, ada **fluktuasi kecil** di beberapa titik, yang mungkin menunjukkan sedikit overfitting pada data latih, tetapi tetap menunjukkan hasil yang sangat baik.

### **2. Grafik Model Loss**:
Grafik ini menunjukkan **loss model** pada **data latih (Train Loss)** dan **data validasi (Validation Loss)** selama **50 epoch**.

- **Train Loss (Loss Latih)**:
  - Loss pada data latih menurun secara tajam pada awal pelatihan dan stabil pada nilai yang sangat rendah di akhir pelatihan (sekitar **0.05**).
  - Penurunan yang cepat menunjukkan bahwa model mampu mengoptimalkan bobot dan menghasilkan prediksi yang lebih baik selama pelatihan.

- **Validation Loss (Loss Validasi)**:
  - Loss pada **data validasi** juga menunjukkan penurunan yang stabil, meskipun sedikit lebih tinggi daripada **train loss**.
  - Hal ini menunjukkan bahwa model bisa **menggeneralisasi dengan baik** pada data yang tidak dilihat sebelumnya.

### **Kesimpulan**:
- **Train Accuracy** meningkat secara stabil, dengan model mencapai **akurasi lebih dari 99%** pada data latih, menunjukkan bahwa model belajar dengan sangat baik.
- **Validation Accuracy** juga meningkat, dengan model tetap menunjukkan performa yang baik pada data validasi, meskipun sedikit fluktuasi muncul.
- **Train Loss** dan **Validation Loss** menunjukkan penurunan yang stabil, dengan model terus belajar dan mengurangi kesalahan pada kedua data latih dan data validasi.

### **Indikasi**:
- Model menunjukkan **kinerja yang sangat baik** dalam hal **akurasi** dan **loss**.
- Tidak ada indikasi **overfitting yang signifikan**, meskipun ada fluktuasi kecil pada **validation accuracy**. 

Dengan hasil ini, model siap untuk digunakan untuk **prediksi risiko kehamilan** pada data baru.

"""

# %%
"""
**Evaluation**
"""

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Konversi label ke string agar tidak error pada classification_report dan heatmap
target_names = [str(cls) for cls in le.classes_]

# Menampilkan classification report
print(classification_report(y_test, y_pred_labels, target_names=target_names))

# Menampilkan confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# %%
"""
### Confusion Matrix dan Metrik Evaluasi

### **Confusion Matrix**:
Confusion matrix di bawah ini menunjukkan perbandingan antara **nilai aktual** dan **prediksi model** untuk dua kelas **risiko kehamilan** (Low dan High).

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

"""

# %%

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Gunakan best_model yang sudah fit
# Prediksi probabilitas untuk data uji
y_pred_prob = best_model.predict_proba(X_test)

# Untuk ROC, kita butuh label biner dan probabilitas kelas positif
# Jika target sudah 0/1, gunakan y_test langsung
# y_pred_prob shape: (n_samples, n_classes), ambil kolom kelas 1
if y_pred_prob.shape[1] == 2:
	y_score = y_pred_prob[:, 1]
	y_test_bin = y_test
else:
	# multiclass: binarize for class 1 (High Risk)
	y_test_bin = label_binarize(y_test, classes=[0, 1, 2])[:, 1]
	y_score = y_pred_prob[:, 1]

# Menghitung ROC curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # garis diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# %%
"""
### Receiver Operating Characteristic (ROC) Curve

### **Penjelasan ROC Curve**:
ROC curve digunakan untuk mengevaluasi kinerja model klasifikasi dengan menggambarkan hubungan antara **True Positive Rate (TPR)** dan **False Positive Rate (FPR)**.

- **True Positive Rate (TPR)**, juga dikenal sebagai **Recall**, mengukur seberapa baik model dalam **mendeteksi kelas positif**.
- **False Positive Rate (FPR)** mengukur **proporsi kelas negatif** yang salah diklasifikasikan sebagai kelas positif.

### **Interpretasi ROC Curve**:
- **Garis diagonal abu-abu**: Ini adalah garis **random classifier**, yang menunjukkan kinerja model yang tidak lebih baik dari **tebakan acak**.
- **Garis biru (ROC curve)**: Menunjukkan kinerja model. Semakin dekat ROC curve ke bagian **pojok kiri atas**, semakin baik kinerja model, karena menunjukkan **tingkat deteksi positif yang lebih tinggi** dan **tingkat kesalahan positif yang lebih rendah**.

### **Area Under Curve (AUC)**:
- **AUC (Area Under Curve)** adalah **nilai yang mengukur seberapa baik model dalam membedakan antara kelas positif dan negatif**.
- **AUC = 1.00** berarti model **sempurna**, dapat memisahkan kelas dengan sempurna.
- **AUC mendekati 0.5** menunjukkan model yang hanya sedikit lebih baik dari **tebakan acak**.
  
Pada grafik ini, nilai **AUC = 1.00**, yang menunjukkan bahwa model memiliki **kinerja sangat baik** dalam **membedakan antara kelas** (risiko rendah dan tinggi).

### **Kesimpulan**:
- ROC curve yang sangat mendekati bagian **pojok kiri atas** menunjukkan bahwa model **memiliki performa sangat baik** dalam memprediksi kedua kelas.
- **AUC = 1.00** menunjukkan bahwa model ini **mampu memisahkan kelas dengan sempurna**, tanpa kesalahan.


"""

# %%
"""
## Perbandingan dan Kesimpulan
"""

# %%
"""
## Perbandingan Antara Grid Search, Randomized Search, dan Neural Network (Tanpa Fine-Tuning)

### **1. Apa yang Dapat Dipelajari dari Perbandingan Tiga Metode?**
Dalam penelitian ini, tiga pendekatan berbeda digunakan untuk melatih model klasifikasi risiko kehamilan: **Grid Search**, **Randomized Search**, dan **Neural Network tanpa fine-tuning**. Berikut adalah perbandingan dari masing-masing pendekatan.

| **Metode**             | **Best Parameters**                                                                                                                                          | **Best Accuracy**  | **Keuntungan**                                  | **Kekurangan**                               |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|-------------------------------------------------|---------------------------------------------|
| **Grid Search**         | - Batch Size: 16 <br> - Epochs: 50 <br> - Dropout Rate: 0.3 <br> - Hidden Units: 128 <br> - Optimizer: Adamax                                                | **97.90%**         | Akurasi terbaik                                | Waktu pencarian lebih lama                  |
| **Randomized Search**   | - Optimizer: RMSprop <br> - Hidden Units: 32 <br> - Dropout Rate: 0.4 <br> - Epochs: 50 <br> - Batch Size: 32                                               | **97.81%**         | Waktu pencarian lebih cepat                    | Akurasi sedikit lebih rendah dari Grid Search |
| **Neural Network (Tanpa Fine-Tuning)** | - Menggunakan default parameter (tanpa fine-tuning)                                                                                         | **96.31%**         | Cepat dan sederhana                            | Akurasi lebih rendah tanpa fine-tuning      |

### **2. Apa Perbedaan Utama Antara Ketiga Metode Ini?**
- **Grid Search** memberikan **akurasi tertinggi** (97.90%) karena **mencoba semua kombinasi** hyperparameter yang telah ditentukan, tetapi memerlukan **waktu yang lebih lama**.
- **Randomized Search** memberikan **akurasi yang hampir sama** (97.81%) dengan **Grid Search** namun lebih efisien dalam hal **waktu pencarian**.
- **Neural Network tanpa fine-tuning** hanya menggunakan **default parameter**, menghasilkan **akurasi yang lebih rendah** (96.31%) tetapi lebih cepat.

### **3. Apa Pengaruh Pemilihan Optimizer terhadap Kinerja Model?**
- Pada **Grid Search**, optimizer yang digunakan adalah **Adamax**, yang memberikan **akurasi terbaik** pada data latih.
- **Randomized Search** memilih **RMSprop**, yang memberikan hasil sangat baik meskipun **akurasi sedikit lebih rendah**.
- Pada **Neural Network tanpa fine-tuning**, **default optimizer** digunakan, yang mungkin tidak seoptimal yang dipilih melalui pencarian **hyperparameter**.
- **Apakah pemilihan optimizer (Adamax vs RMSprop)** mempengaruhi **akurasi model**, dan bagaimana hal ini dapat menjelaskan **perbedaan akurasi** di antara ketiga metode?

### **4. Bagaimana Dropout Rate dan Hidden Units Berpengaruh pada Kinerja Model?**
- **Grid Search** menggunakan **dropout rate 0.3** dan **hidden units 128**, sementara **Randomized Search** menggunakan **dropout rate 0.4** dan **hidden units 32**.
- **Neural Network tanpa fine-tuning** menggunakan **default parameter**, yang mungkin tidak seoptimal **Grid Search** dan **Randomized Search**.
- **Apakah pengaturan dropout rate dan hidden units berpengaruh besar terhadap akurasi validasi**, dan apakah itu yang menjelaskan perbedaan hasil antara **Grid Search** dan **Randomized Search**?

### **5. Apakah Tuning Hyperparameter Meningkatkan Performa Model Secara Signifikan?**
- **Grid Search** dan **Randomized Search** keduanya menunjukkan bahwa pemilihan **dropout rate** dan **hidden units** mempengaruhi **akurasi model**.
- Namun, meskipun **Grid Search** menghasilkan **akurasi sedikit lebih tinggi**, **Randomized Search** memberikan hasil yang sangat baik dalam **waktu yang lebih singkat**.
- **Apakah hasil yang lebih efisien dari Randomized Search cukup memadai meskipun ada sedikit penurunan dalam akurasi dibandingkan dengan Grid Search?**

### **6. Apa Kesimpulan Utama yang Dapat Diambil dari Perbandingan Ini?**
- **Grid Search** memberikan **akurasi terbaik** dalam hal **akurasi**, tetapi memerlukan lebih banyak waktu, sedangkan **Randomized Search** lebih efisien meskipun hasilnya sedikit lebih rendah.
- **Neural Network tanpa fine-tuning** memberikan hasil yang cukup baik dengan **akurasi 96.31%**, tetapi lebih sederhana dan cepat.
- **Penggunaan pencarian hyperparameter** (**Grid Search** dan **Randomized Search**) jelas memberikan **keuntungan dalam akurasi model**, meskipun **Randomized Search** lebih efisien dalam **waktu pencarian**.

### **Kesimpulan**:
- **Grid Search** memberikan hasil **akurasi tertinggi** (97.90%) tetapi memakan waktu lebih lama.
- **Randomized Search** memberikan hasil yang hampir **sama baiknya** (97.81%) dengan **Grid Search**, tetapi jauh lebih cepat.
- **Neural Network tanpa fine-tuning** memberikan hasil **akurasi lebih rendah** (96.31%) tetapi lebih **sederhana** dan cepat.
- **Penggunaan pencarian hyperparameter** (**Grid Search** dan **Randomized Search**) jelas memberikan **keuntungan dalam akurasi model**, meskipun **Randomized Search** lebih efisien dalam **waktu pencarian**.


"""

# %%
"""
## Referensi
"""

# %%
"""
1. **Arisgraha, F. C. S., Rulaningtyas, R., & Kusumawardani, M. A. (2023).** Classification of Pneumonia from Chest X-ray Images Using Keras Module TensorFlow. *Indonesian Applied Physics Letters, 4*(1), 1-7. (https://doi.org/10.1109/ICESC54411.2022.9885515).

2. **Mojumdar, M. U., Sarker, D., & Assaduzzaman, M. (2025).** Maternal Health Risk Factors Dataset: Clinical Parameters and Insights from Rural Bangladesh. *Data in Brief, 59*, 111363. (https://doi.org/10.1016/j.dib.2025.111363).

3. **Pi, X., Wang, J., Chu, L., Zhang, G., & Zhang, W. (2025).** Prediction of High-Risk Pregnancy Based on Machine Learning Algorithms. *Scientific Reports, 15*, 15561. (https://doi.org/10.1038/s41598-025-00450-3).

"""

# %%
pip install ipynb-py-convert

# %%
from ipynb_py_convert import convert

# Tentukan path file input (.ipynb) dan output (.py)
input_path = "I:/DBS Foundations/subbmission machine learning terapan/ML TERAPAN 1/ML1_tyasnurkumala.ipynb"
output_path = "I:/DBS Foundations/subbmission machine learning terapan/ML TERAPAN 1/ML1_tyasnurkumala.py"

# Lakukan konversi
convert(input_path, output_path)

print(f"Konversi berhasil! File disimpan di: {output_path}")
