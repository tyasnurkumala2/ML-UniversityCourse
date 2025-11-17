from utils.extract import scrape_all_pages
from utils.transform import transform_to_DataFrame, transform_data
from utils.load import store_to_csv, store_to_gsheet

def main():
    print("[ETL] Memulai proses pengambilan data...")
    raw_data = scrape_all_pages()

    if raw_data:
        try:
            print("[ETL] Mulai proses pembersihan dan transformasi data...")
            df_raw = transform_to_DataFrame(raw_data)
            df_clean = transform_data(df_raw, exchange_rate=16000)

            print("[ETL] Menyimpan hasil ke file CSV dan Google Sheets...")
            store_to_csv(df_clean, filename="products.csv")

            # Gunakan ID spreadsheet secara langsung
            spreadsheet_id = "1t9_iKLQ4vwupN7DNgbyoPpl6jre0sQK-DCAYvWfVQL8"

            if store_to_gsheet(
                df_clean,
                spreadsheet_id=spreadsheet_id,
                json_keyfile_name="google-sheets-api.json",  # Nama file JSON API yang digunakan
                sheet_name="product"  # Nama sheet yang dituju
            ):
                print("[INFO] Data berhasil disimpan ke Google Sheets.")
            else:
                print("[ERROR] Gagal mengirimkan data ke Google Sheets.")

            print("[ETL] Proses selesai. Data telah berhasil diproses dan disimpan.")
            print(df_clean.head())

        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat pembersihan atau penyimpanan data: {e}")
    else:
        print("⚠️ [ETL] Tidak ada data yang berhasil diambil untuk diproses.")

if __name__ == "__main__":
    main()
