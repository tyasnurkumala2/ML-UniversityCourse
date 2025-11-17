import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Menambahkan direktori utils ke sys.path agar modul load bisa diimpor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import store_to_csv, store_to_gsheet

class LoadFunctionsTestCase(unittest.TestCase):

    @patch('utils.load.pd.DataFrame.to_csv')
    def test_store_to_csv_function(self, mock_to_csv):
        sample_data = pd.DataFrame({'Title': ['Produk 1'], 'Price': [200]})
        store_to_csv(sample_data, "hasil_output.csv")
        mock_to_csv.assert_called_once_with("hasil_output.csv", index=False)

    @patch('utils.load.gspread.authorize')
    @patch('utils.load.Credentials')
    def test_store_to_google_sheet(self, mock_creds_class, mock_authorize_func):
        # Mock credentials dan client gspread
        fake_creds = MagicMock()
        mock_creds_class.from_service_account_file.return_value = fake_creds

        fake_client = MagicMock()
        mock_authorize_func.return_value = fake_client

        fake_spreadsheet = MagicMock()
        fake_worksheet = MagicMock()

        # Setup untuk worksheet bernama 'product'
        fake_spreadsheet.worksheet.return_value = fake_worksheet
        fake_client.open_by_key.return_value = fake_spreadsheet

        # Data dummy untuk diuji
        data_frame = pd.DataFrame({'Title': ['Produk 1'], 'Price': [200]})

        # Jalankan fungsi yang diuji
        store_to_gsheet(
            data_frame,
            spreadsheet_id="1t9_iKLQ4vwupN7DNgbyoPpl6jre0sQK-DCAYvWfVQL8",
            json_keyfile_name="google-sheets-api.json"
        )

        # Assertion pengecekan pemanggilan fungsi internal
        mock_creds_class.from_service_account_file.assert_called_once_with(
            "google-sheets-api.json",
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        mock_authorize_func.assert_called_once_with(fake_creds)
        fake_client.open_by_key.assert_called_once_with("1t9_iKLQ4vwupN7DNgbyoPpl6jre0sQK-DCAYvWfVQL8")
        fake_spreadsheet.worksheet.assert_called_once_with('product')
        fake_worksheet.clear.assert_called_once()
        fake_worksheet.update.assert_called_once()

if __name__ == "__main__":
    unittest.main()
