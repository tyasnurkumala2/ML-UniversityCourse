import unittest
from unittest.mock import patch, Mock
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os

# Menambahkan folder root proyek ke sys.path agar modul utils bisa diimpor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.extract import fetching_content, extract_product_data, scrape_all_pages

class TestExtractFunctions(unittest.TestCase):

    @patch('utils.extract.requests.get')
    def test_fetching_content_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Mocked Page</body></html>"
        mock_get.return_value = mock_response

        result = fetching_content("https://example.com")
        self.assertEqual(result, "<html><body>Mocked Page</body></html>")

    @patch('utils.extract.requests.get')
    def test_fetching_content_failure(self, mock_get):
        mock_get.side_effect = Exception("Request failed")
        result = fetching_content("https://example.com")
        self.assertIsNone(result)

    def test_extract_product_data_all_valid(self):
        html = """
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">T-shirt 2</h3>
                <div class="price-container">
                    <span class="price">$123.20</span>
                </div>
                <p style="font-size: 14px;">Rating: ⭐ 4.8</p>
                <p style="font-size: 14px;">Colors: 3 Colors</p>
                <p style="font-size: 14px;">Size: M</p>
                <p style="font-size: 14px;">Gender: Women</p>
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        item = soup.find("div", class_="collection-card")
        result = extract_product_data(item)
        result.pop("Timestamp")

        expected = {
            "Title": "T-shirt 2",
            "Price": 123.20,
            "Rating": "⭐ 4.8",
            "Colors": "3 Colors",
            "Size": "M",
            "Gender": "Women"
        }
        self.assertEqual(result, expected)

    def test_extract_product_data_invalid_rating(self):
        html = """
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">Unknown Product</h3>
                <div class="price-container">
                    <span class="price">$10.00</span>
                </div>
                <p style="font-size: 14px;">Rating: ⭐ Invalid Rating</p>
                <p style="font-size: 14px;">Colors: 5 Colors</p>
                <p style="font-size: 14px;">Size: M</p>
                <p style="font-size: 14px;">Gender: Men</p>
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        item = soup.find("div", class_="collection-card")
        result = extract_product_data(item)
        result.pop("Timestamp")

        expected = {
            "Title": "Unknown Product",
            "Price": 10.00,
            "Rating": "⭐ Invalid Rating",
            "Colors": "5 Colors",
            "Size": "M",
            "Gender": "Men"
        }
        self.assertEqual(result, expected)

    def test_extract_product_data_not_rated(self):
        html = """
        <div class="collection-card">
            <div class="product-details">
                <h3 class="product-title">Pants 16</h3>
                <p class="price">Price Unavailable</p>
                <p style="font-size: 14px;">Rating: Not Rated</p>
                <p style="font-size: 14px;">Colors: 3 Colors</p>
                <p style="font-size: 14px;">Size: XL</p>
                <p style="font-size: 14px;">Gender: Men</p>
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        item = soup.find("div", class_="collection-card")
        result = extract_product_data(item)
        result.pop("Timestamp")

        expected = {
            "Title": "Pants 16",
            "Price": None,
            "Rating": "Not Rated",
            "Colors": "3 Colors",
            "Size": "XL",
            "Gender": "Men"
        }
        self.assertEqual(result, expected)

    @patch('utils.extract.fetching_content')
    def test_scrape_all_pages_empty(self, mock_fetch):
        mock_fetch.return_value = "<html><body>No products</body></html>"
        result = scrape_all_pages()
        self.assertEqual(result, [])

    @patch('utils.extract.time.sleep', return_value=None)
    @patch('utils.extract.fetching_content')
    def test_scrape_all_pages_mocked(self, mock_fetch, mock_sleep):
        page_1_html = """
        <html>
            <body>
                <div class="collection-card">
                    <h3 class="product-title">Mocked Shirt</h3>
                    <span class="price">$25.00</span>
                    <p style="font-size: 14px;">Rating: 4.0</p>
                    <p style="font-size: 14px;">Colors: Blue</p>
                    <p style="font-size: 14px;">Size: L</p>
                    <p style="font-size: 14px;">Gender: Men</p>
                </div>
            </body>
        </html>
        """
        empty_html = "<html><body>No products</body></html>"
        mock_fetch.side_effect = [page_1_html] + [empty_html] * 49

        results = scrape_all_pages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['Title'], 'Mocked Shirt')
        self.assertEqual(results[0]['Price'], 25.00)
        self.assertEqual(results[0]['Size'], 'L')

# Bagian kode di bawah ini adalah contoh pengujian utama yang saat ini dikomentari
# yang akan menguji seluruh alur utama dari scraping sampai penyimpanan data.
# Jika diperlukan, Anda dapat mengaktifkannya dengan menghilangkan tanda komentar.

#     @patch('utils.extract.store_to_postgre')
#     @patch('utils.extract.store_to_csv')
#     @patch('utils.extract.store_to_gsheet')
#     @patch('utils.extract.transform_to_DataFrame')
#     @patch('utils.extract.transform_data')
#     @patch('utils.extract.scrape_all_pages')
#     def test_main_success(
#         self, mock_scrape, mock_transform, mock_df, mock_gsheet, mock_csv, mock_pg
#     ):
#         # Menyiapkan data tiruan untuk pengujian
#         mock_data = [{
#             "Title": "Mock", "Price": 100.0, "Rating": "4.5", "Colors": "Red",
#             "Size": "L", "Gender": "Unisex", "Timestamp": "2024-01-01 00:00:00"
#         }]
#         df_mock = pd.DataFrame(mock_data)
#         mock_scrape.return_value = mock_data
#         mock_df.return_value = df_mock
#         mock_transform.return_value = df_mock

#         # Menjalankan fungsi utama
#         main()

#         # Memastikan setiap fungsi dalam alur utama terpanggil dengan benar
#         mock_scrape.assert_called_once()
#         mock_df.assert_called_once()
#         mock_transform.assert_called_once()
#         mock_pg.assert_called_once()
#         mock_csv.assert_called_once()
#         mock_gsheet.assert_called_once()

# if __name__ == "__main__":
#     unittest.main()
