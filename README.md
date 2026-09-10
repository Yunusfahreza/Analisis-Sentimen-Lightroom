# Analisis Sentimen Ulasan Pengguna — Adobe Lightroom Mobile

Analisis sentimen terhadap ulasan pengguna aplikasi **Adobe Lightroom Mobile** di Google Play Store (ulasan berbahasa Indonesia). Karena dataset tidak memiliki label sentimen manual, label dibuat secara otomatis menggunakan model **IndoBERT** yang sudah dilatih sebelumnya (pre-trained), lalu beberapa model klasifikasi dilatih dan dibandingkan di atas label tersebut.

## Alur Notebook

1. **Setup & Konfigurasi** — import library dan parameter utama dikumpulkan di satu tempat.
2. **Pengumpulan Data (Scraping)** — mengambil ulasan dari Google Play Store menggunakan `google_play_scraper`, dengan pagination (`continuation_token`) dan retry sederhana untuk error jaringan.
3. **Preprocessing Teks** — cleaning minimal (normalisasi huruf, hapus URL, perbaikan pengulangan huruf seperti `bagusss` → `bagus`, dan sebagian slang umum), tanpa stemming atau stopword removal secara sengaja, karena IndoBERT butuh struktur kalimat dan tanda baca untuk memahami konteks.
4. **Pelabelan Otomatis dengan IndoBERT** — label sentimen dihasilkan dari model sentimen Bahasa Indonesia pre-trained (`w11wo/indonesian-roberta-base-sentiment-classifier`, dengan fallback model lain jika gagal diunduh). Kualitas label bergantung pada model ini, bukan anotasi manusia.
5. **Penyeimbangan Kelas** — oversampling kelas minoritas dan undersampling kelas mayoritas menuju target yang sama per kelas (bukan SMOTE, karena data masih berupa teks mentah).
6. **Split Data & Ekstraksi Fitur TF-IDF** — split 70% train / 10% validation / 20% test (stratified), representasi teks dengan TF-IDF (unigram–trigram).
7. **Model Machine Learning** — Naive Bayes, SVM, dan Random Forest, di-tuning dengan `GridSearchCV` (cross-validation 3-fold).
8. **Model Deep Learning (BiLSTM)** — menangkap konteks kalimat dari dua arah, berguna untuk negasi dan konteks yang sering muncul di ulasan Bahasa Indonesia.
9. **Evaluasi & Analisis Overfitting** — membandingkan akurasi training vs validation, serta confusion matrix BiLSTM untuk melihat kelas yang paling sering tertukar.
10. **Ensemble Model** — menggabungkan prediksi Naive Bayes, SVM, Random Forest, dan BiLSTM dengan *weighted voting* (bobot lebih besar untuk SVM dan Random Forest).
11. **Ringkasan Hasil** — statistik dan metrik dihitung langsung dari hasil run (bukan angka tetap; berubah tiap kali notebook dijalankan ulang tergantung data yang ter-scrape).
12. **Word Cloud** — visualisasi kata yang paling sering muncul secara umum maupun per kelas sentimen.
13. **Demo Prediksi** — fungsi `predict_and_show("kalimat kamu")` untuk mencoba ensemble pada kalimat baru.

## Model yang Digunakan

| Tahap | Model |
|---|---|
| Pelabelan otomatis | IndoBERT (RoBERTa Bahasa Indonesia, pre-trained) |
| Machine Learning | Naive Bayes, SVM, Random Forest (TF-IDF) |
| Deep Learning | BiLSTM |
| Final | Ensemble weighted voting (NB 0.2, SVM 0.3, RF 0.3, BiLSTM 0.2) |

## Catatan Penting

- **Label bukan ground truth manusia.** Label sentimen dihasilkan oleh model IndoBERT pre-trained, sehingga akurasi model akhir ikut dibatasi oleh kualitas model pelabel tersebut. Validasi manual pada sebagian sampel disarankan untuk kebutuhan yang lebih serius.
- **Hasil scraping tidak deterministik.** Karena data diambil langsung dari Google Play Store saat notebook dijalankan, jumlah dan isi ulasan — serta seluruh metrik di bagian Ringkasan Hasil — bisa berbeda setiap kali notebook di-run ulang.
- Untuk memakai model ensemble di luar notebook, tidak disarankan meng-`pickle` seluruh objek `EnsembleSentimentAnalyzer` (berisiko untuk komponen Keras di dalamnya). Cukup simpan dan load ulang komponennya secara terpisah: `tfidf` dan model ML lewat `joblib`, serta `model_bilstm` lewat `tf.keras.models.load_model`, lalu bentuk ulang instance ensemble-nya.

## Cara Menjalankan

```bash
pip install pandas numpy scikit-learn tensorflow torch transformers \
            google-play-scraper wordcloud matplotlib seaborn tqdm joblib
```

Lalu jalankan notebook `analisisbertlr_simplified.ipynb` dari atas ke bawah. Proses scraping dan pelabelan IndoBERT membutuhkan koneksi internet dan bisa memakan waktu tergantung jumlah ulasan target (`TARGET_REVIEWS`) dan ketersediaan GPU untuk mempercepat inferensi IndoBERT.

## Struktur Repo

```
.
├── analisisbertlr_simplified.ipynb
└── README.md
```

## Tools

Python, PyTorch, Transformers (IndoBERT), TensorFlow/Keras (BiLSTM), scikit-learn, google-play-scraper, WordCloud
