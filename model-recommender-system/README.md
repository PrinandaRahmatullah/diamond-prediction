# Laporan Proyek Machine Learning - Prinanda Rahmatullah

## Domain Proyek
Dari sumber data yang diperoleh di Kaggle, terdapat konteks yang menceritakan tentang Bob sudah mendirikan perusahan ponselnya sendiri. Dia ingin memberikan persaingan yang sengit terhadap beberapa perusahaan ponsel besar seperti Apple, Samsung, dan lainnya. Bob tidak tau bagaimana cara mengestimasi harga dari ponsel yang dibuat oleh perusahaannya. Di masa saat ini dimana pasar ponsel yang sangat kompetitif,kita tidak bisa menyimpulkan sesuatu secara sederhana. Untuk menyelesaikan perkara ini, dia mengumpulkan data penjualan ponsel dari beberapa perusahaan. Bob juga ingin mengetahui relasi dari fitur-fitur ponsel yang ada dari data yang dia kumpulkan dan hubungan fitur tersebut dengan kelompok harga ponsel. Tetapi, dia tidak terlalu mengerti tentang Machine Learning. Jadi, dia meminta saya selaku ML Engineer lulusan Dicoding Indonesia untuk membantunya mengatasi masalah ini.

## Business Understanding
### Problem Statements
- Bagaimana cara mengetahui dan menemukan fitur-fitur apa saja dari data yang dapat mempengaruhi rentang harga ponsel?
- Apakah ada cara untuk mendapat estimasi rentang harga jual ponsel dari data menggunakan model Machine Learning?

### Goals
Tujuan utamanya adalah : 
- Mengetahui fitur-fitur apa saja yang mempengaruhi rentang harga ponsel.
- Membangun model sederhana untuk memprediksi atau estimasi rentang harga ponsel. 

### Solution Statement.
- Solusi untuk problem dan goals yang pertama adalah dengan pendekatan Exploratory Data Analysis
- Solusi untuk problem dan goals yang kedua adalah dengan membangun model Supervised Learning seperti KNN, Random Forest, dan Boosting. Performa modelnya akan diukur menggunakan metrics yang cocok dengan Categorical Classification yaitu F1-score yang terdiri atas Precision, Recall, dan F1-score itu sendiri.

## Data Understanding
Sumber data : https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

Data harga ponsel terdiri atas dua file yaitu train.csv dan test.csv. File diload ke dalam masing-masing dataframe. Dataframe train terdiri atas 2000 baris data dan 21 kolom termasuk rentang harga, sedangakan Dataframe test terdiri atas 1000 baris data dan 20 kolom karena data test tidak memiliki kolom rentang harga.

### Variabel-variabel pada Data Mobile Price
Data columns (total 21 columns):
1. battery_power = Total energi yang dapat disimpan oleh sebuah baterai dalam satuan mAh.
2. blue = Ada atau tidaknya bluetooth. | 1: ada , 0: tidak ada.
3. clock_speed = Kecepatan atau frekuensi mikroprosesor mengeksekusi intruksi.
4. dual_sim = Mendukung dual sim atau tidak. 1: mendukung , 0: tidak mendukung
5. fc = Resolusi kamera depan dalam satuan mega piksel.
6. four_g = Mendukung 4G atau tidak. | 1: mendukung , 0: tidak mendukung
7. int_memory = Kapasitas memori internal dalam satuan GigaBytes (GB).
8. m_dep = Ketebalan ponsel dalam satuan cm.
9. mobile_wt = Berat atau massa ponsel.
10. n_cores = Jumlah prosesor.
11. pc = Resolusi kamera utama atau belakang dalam satuan mega piksel.
12. px_height = Resolusi tinggi layar dalam satuan piksel.
13. px_width = Resolusi lebar layar dalam satuan piksel.
14. ram = Besar Random Access Memory dalam satuan GigaBytes (GB).
15. sc_h = Tinggi ponsel dalam satuan cm.
16. sc_w = Lebar ponsel dalam satuan cm.
17. talk_time = Lama kemampuan bertahan baterai dalam sekali pengisian daya.
18. three_g = Mendukung 3G atau tidak. | 1: mendukung , 0: tidak mendukung
19. touch_screen = Mendukung layar sentuk atau tidak. | 1: mendukung , 0: tidak mendukung
20. wifi = Mendukung wifi atau tidak. | 1: mendukung , 0: tidak mendukung
21. price_range = Kelompok rentang harga sebagai target variabel atau kelas data. | 3: Pricey, 2: Expensive, 1: Medium, 0: Cheap


### DataFrame.describe()
Perintah ini menginformasikan data statistik dari masing-masing kolom berupa jumlah baris, nilai rata-rata, nilai minimum & maksimum, kuartil, dan standar deviasi

### Null Data dan Zero Value
Selanjutnya pada EDA adalah dalam mengecek data null atau kosong menggunakan command train_df.isna().sum() atau train_df.isnull().sum(). Kemudian berdasarkan hasil DataFrame.describe() di atas, diketahui bahwa ada kolom/fitur ponsel yang seharusnya nilai terendahnya bukan 0 (zero) melainkan suatu angka. Kolom tersebut adalah Pixel Height dan Screen Width. Oleh karena itu, baris data yang memiliki nilai 0 pada kolom tersebut harus dihapus dengan menggunakan perintah DataFrame.drop() agar tidak mempengaruhi kesimpulan eksplorasi data dan pengembangan model nantinya.

### Count Each Price Range
Pada data asli, price range atau rentang harga ponsel menggunakan data kelompok angka. Di sini, saya mengubah data tersebut menjadi nama agar memudahkan pembaca memahaminya. Rentang harga terbagi atas 4 kategori yaitu Cheap 24.8%, Medium 24.8%, Expensive 25.1%, dan Pricey 25.3%. <br>
![Mobile Phone Price Range](images/price_range.png "Mobile Phone Price Range")

### Feature Correlation
Tujuan pertama yang diinginkan oleh Bob adalah melihat fitur ponsel mana yang memiliki hubungan terhadap harga ponsel. Dengan command DataFrame.corr(), kita dapat mengetahui hubungan tersebut seperti yang terdapat pada gambar berikut. Skor korelasi berkisar dari -1 dan 1. Skor yang semakin jauh dari angka 0, semakin kuat hubungan terhadap harga. Jika skor mendekati -1, fitur tersebut memiliki efek terbalik dengan harga. Semakin rendah nilai pada kolom tersebut, semakin tinggilah harga ponsel. Akan tetapi, jika skor mendekati angak 1, fitur tersebut memiliki efek selaras dengan harga. Semakin tinggi nilai pada kolom tersebut, semkain tinggilah harga ponsel. Kemudian jika mendekati 0, kolom tersebut memiliki hubungan yang lemah terhadap harga ponsel. Di sini kita dapat mengetahui bahwa:
- Jumlah RAM sangat erat memengaruhi harga ponsel dengan skor korelasi 0.9
- Battery power, pixel width, and pixel height memliki hubungan yang lemah terhadap harga ponsel dengan skor korelasi antara 0.1 and 0.2
- Fitur yang lainnya tidak terlihat memiliki hubungan terhadap harga ponsel
- Primary camera megapixel (pc) memiliki efek yang cukup erat terhadap Front camera mega pixel (fc) dengan skor korelasi 0.6
- Three G (three_g) memiliki efek yang cukup erat terhadap four G (four_g) dengan skor korelasi 0.6
- Pixel width (px_width) memiliki efek yang cukup erat terhadap pixel height (px_height) dengan skor korelasi 0.5
- Screen width (px_width) memiliki efek yang cukup erat terhadap screen height (px_height) dengan skor korelasi 0.5

![Mobile Feature-Price Correlation](images/correlation.png "Mobile Feature-Price Correlation")

## Data Preparation

### Load Data
Load data train dan test dari dataframe sebelumnya. Untuk dataframe X sebagai data latih, kita harus membuang kolom price_range karena kolom tersebut adalah kelas data. Kemudian y adalah subset dari Dataframe train yang berisi kolom price_range saja. Sedangkan dataframe X_test hanya perlu diassign dari Dataframe test.

### Standardize Data
Sebenarnya data bisa dinormalisasi atau standardisasi agar nilai tidak terlalu besar ketika dikalkulasi dengan algoritma machine learning menggunakan StandardScaler atau MinMaxScaler, tetapi pada submission ini tidak saya gunakan karena hasil sudah baik dan tidak terganggu oleh besarnya nilai pada fitur-fitur yang ada.

### Split Data
Untuk memudahkan proses evaluasi performa model, kita perlu membagi data X menjadi train dan validation menggunakan train_test_split() dari scikit-learn. Data dibagi menjadi 70% training dan 30% validasi.


## Modeling
Model machine learning yang dipilih adalah Supervised Machine Learning ranah classification karena tipe kelas data yang disediakan oleh dataset temasuk kategori. Oleh karena itu, classification adalah pilihan utamanya. Algoritma yang dipilih adalah 3 algoritma yang diajarkan pada kelas ML Terapan yaitu K-Nearest Neighbors, Random Forest, dan AdaBoost. Awalnya saya ingin menggunakan semua algoritma classification traditional machine learning dan neural network. Berhubung saya sedang sibuk bekerja, project, dan susah membagi waktu, saya gunakan yang ada di modul saja untuk menghemat waktu.

### KNN
KNN belajar dari data terhadap tetangga terdekatnya sebanyak k. Untuk menentukan nilai k, kita bisa melakukan iterasi sekaligus mencari f1-score terbaik dari k tersebut. Dari gambar di bawah, terlihat bahwa k terbaik adalah 21 dengan f1-score 0.93.
![KNN Mobile Price-Range Prediction](images/knn.png "KNN Mobile Price-Range Prediction")

### Random Forest
Random forest belajar dari data kumpulan pohon yang menjadi hutan. Konsep ini adalah perkembangan dari decision tree dengan parameter yaitu n estimator dan depth. Di sini, saya menerapkan fixed depth agar komputasi tidak terlalu tinggi yaitu 8 max depth. Namun N estimator tetap harus dicari agar mendapatkan akurasi terbaik. Dari gambar di bawah, terlihat bahwa n estimator terbaik adalah 140 dengan f1-score 0.90.
![Random Forest Mobile Price-Range Prediction](images/rf.png "Random Forest Mobile Price-Range Prediction")

### Boosting
Boosting menggunakan weak classifier kemudian diatur learning rate beserta n-estimator untuk mencari best performance model. Di sini saya menggunakan GridSearchC Cross Validation untuk menemukan learning rate dan n-estimator terbaik. Diperoleh bahwa learning rate terbaik adalah 0.05 dan n-estimator 42 dengan f1-score 0.73.

## Evaluation
Metrics evaluasi yang digunakan pada kasus ini adalah f1-score. F1-score dipilih karena kasus ini tergolong Categorical Classification. F1-score sendiri merupakan metrics yang dihasilkan dari hubungan beberapa metrics lainnya yaitu precision dan recall yang diperoleh dari perhitungan True Positive, False Positive, dan False Negatif hasil prediksi. Penjelasan detailnya dapat dilihat pada artikel dari scikit-learn https://scikit-learn.org/stable/modules/model_evaluation.html dan jurnal berikut https://www.researchgate.net/publication/226675412_A_Probabilistic_Interpretation_of_Precision_Recall_and_F-Score_with_Implication_for_Evaluation.

### Penjelasan Metrics
- True positive (TP) adalah banyaknya kategori data aktual suatu kelas A yang benar diprediksi oleh model sebagai kelas A.
- True negative (TN) adalah banyaknya kategori data aktual suatu kelas A yang benar diprediksi oleh model sebagai bukan kelas A.
- False positive (FP) adalah banyaknya kategori data aktual bukan kelas A, tetapi diprediksi oleh model sebagai kelas A.
- False negative (FN) adalah banyaknya kategori data aktual kelas A, tetapi diprediksi oleh model sebagai bukan kelas A.
Keempat data di atas dapat diketahui dari confusion matrix hasil klasifikasi.<br><br>
Kemudian, para peneliti mengembangkan suatu metrics yaitu F1-score yang terdiri atas Precision dan Recall dengan tujuan untuk mengetahui seberapa baik persebaran hasil klasifikasi terhadap masing-masing kelas data. Selain hasil masing-masing kelas, F1-score juga dapat mengetahui hasil akhir gabungan seluruh kelas. <br>
- Precision : 
    ```python
    Precision = TP / (TP + FP)
    ```
- Recall : 
     ```python
    Recall = TP / (TP + FN)
    ```
- F1-score :
     ```python
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    ```

### Hasil
Tabel Hasil Klasifikasi:
| Metrics  | KNN  | Random Forest  | Boosting  |
|---|---|---|---|
| F1-score  | 0.930268	| 0.906234  | 0.733186  |


Dari ketiga model di atas, F1-score terbaik dihasilkan oleh algoritma KNN yaitu 0.93 pada data validasi.
<br>
Tujuan pertama sudah diperoleh pada bagian correlation. Kemudian tujuan kedua yaitu mengembangkan model prediksi harga ponsel. Diperoleh KNN sebagai model prediksi harga ponsel terbaik dengan f1-score 0.93. KNN dapat membantu Bob dalam mengestimasi harga ponsel yang ingin dibuat oleh perusahaannya. Kita bisa meningkatkan f1-score dengan menambahkan data lagi, riset, dan ujicoba menggunakan algoritma machine learning lainnya. Tetapi, berhubung waktu tidak memadai, KNN bisa dikatakan sudah cukup baik dalam memprediksi harga ponsel.

## Terima Kasih