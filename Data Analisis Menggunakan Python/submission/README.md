
#  Dashboard Peminjaman Sepeda

Dashboard ini dibuat dengan menggunakan **Streamlit** untuk memvisualisasikan data peminjaman sepeda.

---

##  Setup Environment - Anaconda
```bash
conda create --name bike-ds python=3.9
conda activate bike-ds
pip install -r requirements.txt
```

---

## Setup Environment - Shell/Terminal
```bash
mkdir proyek_bike_dashboard
cd proyek_bike_dashboard
pipenv install
pipenv shell
pip install -r requirements.txt
```

---

## Run Streamlit App
```bash
streamlit run dashboard.py
```

---

## Struktur Direktori
```
SUBMISSION
├── dashboard
│   ├── dashboard.py
│   └── main_data.csv
├── data
│   ├── combined_dataset.csv
│   ├── day.csv
│   └── hour.csv
├── notebook.ipynb
├── Readme.md
├── requirements.txt
└── url.txt

```

---

## Fitur Dashboard
- Menampilkan data peminjaman sepeda berdasarkan **bulan** dan **hari kerja/libur**.
- Visualisasi dengan **line chart** dan **bar chart**.
- Korelasi antara fitur dengan jumlah peminjaman sepeda.

---


