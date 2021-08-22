import unittest
from unittest.case import TestCase
from entry_point import AbaloneClassifier


class TestAbaloneClassifier(unittest.TestCase):

    def test_extract_column_names(self):
        test_classifier = AbaloneClassifier(
            input_data_location="data/test_data.csv")

        test_classifier.extract()

        actual_column_names = test_classifier.raw_data_frame.columns.to_list()
        expected_column_names = test_classifier.all_columns

        self.assertEqual(expected_column_names, actual_column_names)

    def test_extract_df_size(self):
        test_classifier = AbaloneClassifier(
            input_data_location="data/test_data.csv")

        test_classifier.extract()

        actual_df_size = test_classifier.raw_data_frame.shape
        expected_df_size = (30, len(test_classifier.all_columns))

        self.assertEqual(expected_df_size, actual_df_size)


if __name__ == "__main__":
    unittest.main()
