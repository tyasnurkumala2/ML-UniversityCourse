import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils.transform import transform_data, transform_to_DataFrame
import unittest
import pandas as pd

class TransformModuleTests(unittest.TestCase):

    def setUp(self):
        self.sample_data = [
            {'Title': 'Product 1', 'Price': '100', 'Rating': '⭐4.7', 'Colors': '3 Colors', 'Size': 'M', 'Gender': 'Men'},
            {'Title': 'Product 2', 'Price': '123', 'Rating': '⭐4.2', 'Colors': '3', 'Size': 'XL', 'Gender': 'Women'},
            {'Title': 'Product 3', 'Price': None, 'Rating': 'Not Rated', 'Colors': '7 Colors', 'Size': 'S', 'Gender': 'Unisex'}
        ]

    def test_transform_to_dataframe_with_valid_input(self):
        dataframe = transform_to_DataFrame(self.sample_data)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe.shape[0], 3)

    def test_transform_to_dataframe_with_invalid_input(self):
        wrong_input = "bukan list dictionary"
        dataframe = transform_to_DataFrame(wrong_input)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertTrue(dataframe.empty)

    def test_transform_data_correct_conversion(self):
        df = pd.DataFrame(self.sample_data)
        rate = 16000
        transformed_df = transform_data(df, rate)

        self.assertEqual(transformed_df.shape[0], 2)
        self.assertEqual(transformed_df.iloc[0]['Price'], 100 * rate)
        self.assertEqual(transformed_df.iloc[1]['Price'], 123 * rate)
        self.assertIsInstance(transformed_df.iloc[0]['Price'], float)

    def test_transform_data_filter_invalid_rating_and_unknown_titles(self):
        test_df = pd.DataFrame([
            {'Title': 'Unknown Product', 'Price': '100', 'Rating': '⭐4.0', 'Colors': 'Colors1', 'Size': 'M', 'Gender': 'Men'},
            {'Title': 'Product X', 'Price': '125', 'Rating': 'Invalid Rating', 'Colors': 'Colors2', 'Size': 'L', 'Gender': 'Women'}
        ])
        transformed = transform_data(test_df, 1)
        self.assertTrue(transformed.empty)

    def test_transform_data_handle_nonconvertible_colors(self):
        test_df = pd.DataFrame([
            {'Title': 'Product Y', 'Price': '150', 'Rating': '⭐4.7', 'Colors': 'ColorsABC', 'Size': 'S', 'Gender': 'Men'}
        ])
        result = transform_data(test_df, 1)
        self.assertTrue(pd.isna(result.iloc[0]['Colors']))

    def test_transform_data_remove_duplicate_entries(self):
        dup_df = pd.DataFrame([
            {'Title': 'Product Z', 'Price': '90', 'Rating': '⭐4.7', 'Colors': 'Colors1', 'Size': 'S', 'Gender': 'Unisex'},
            {'Title': 'Product Z', 'Price': '90', 'Rating': '⭐4.7', 'Colors': 'Colors1', 'Size': 'S', 'Gender': 'Unisex'}
        ])
        cleaned = transform_data(dup_df, 1)
        self.assertEqual(cleaned.shape[0], 1)

    def test_transform_data_all_nan_values(self):
        nan_df = pd.DataFrame([
            {'Title': 'Product N', 'Price': None, 'Rating': None, 'Colors': None, 'Size': 'M', 'Gender': 'Women'}
        ])
        result = transform_data(nan_df, 1)
        self.assertTrue(result.empty)

if __name__ == "__main__":
    unittest.main()
