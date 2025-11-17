import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def store_to_csv(data: pd.DataFrame, filename: str = "products.csv"):
    try:
        data.to_csv(filename, index=False)
        print(f"[INFO] Data berhasil disimpan ke {filename}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan ke CSV: {e}")

def store_to_gsheet(data: pd.DataFrame, spreadsheet_id: str, json_keyfile_name: str) -> bool:
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_file(json_keyfile_name, scopes=scopes)
        client = gspread.authorize(creds)

        spreadsheet = client.open_by_key(spreadsheet_id)

        # ✅ gunakan worksheet bernama 'product'
        worksheet = spreadsheet.worksheet('product')

        if data.empty:
            print("[WARNING] DataFrame kosong. Tidak ada yang ditulis ke Google Sheets.")
            return False

        worksheet.clear()
        worksheet.update([data.columns.values.tolist()] + data.values.tolist())

        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Google Sheets: {e}")
        return False
