import unittest
from unittest.case import TestCase
from helpers import PipelineHelpers


class TestAbaloneClassifier(TestCase):

    test_data_column_names = [i for i in range(9)]
    raw_df = PipelineHelpers.extract_data(
        "data/test_data.csv", test_data_column_names)

    def test_extract_column_names(self):

        actual_column_names = self.raw_df.columns.to_list()

        self.assertEqual(self.test_data_column_names, actual_column_names)

    def test_extract_df_size(self):

        actual_df_size = self.raw_df.shape
        expected_df_size = (30, len(self.test_data_column_names))

        self.assertEqual(expected_df_size, actual_df_size)


if __name__ == "__main__":

    unittest.main()
