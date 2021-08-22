import unittest
from unittest.case import TestCase

import numpy as np
import pandas as pd

from helpers import PipelineHelpers


class TestPipelineHelpers(TestCase):

    test_data_column_names = [i for i in range(9)]
    raw_df = PipelineHelpers.extract_data(
        "data/test_data.csv", test_data_column_names)

    def test_extract_column_names(self):

        actual_column_names = self.raw_df.columns.to_list()

        self.assertEqual(self.test_data_column_names, actual_column_names)

    def test_extract_df_shape(self):

        actual_df_shape = self.raw_df.shape
        expected_df_shape = (30, len(self.test_data_column_names))

        self.assertEqual(expected_df_shape, actual_df_shape)

    def test_remove_outliers(self):

        total_not_outliers = 100
        max_value = 5
        not_outliers = np.random.randint(max_value, size=total_not_outliers)
        outlier = max_value * 200
        all_values = np.append(not_outliers, outlier)

        input_df = pd.DataFrame(
            {'count': all_values})
        outliers_removed = PipelineHelpers.remove_outliers(input_df, 'count')

        expected_shape = (100, 1)
        actual_shape = outliers_removed.shape

        self.assertEqual(expected_shape, actual_shape)

    def test_encode_text_column(self):
        input_df = pd.DataFrame({'letter': ['a', 'b', 'c'],
                                 'count': [1, 2, 3]})

        encoder = PipelineHelpers.fit_encoder(input_df, 'letter')
        processed_data = PipelineHelpers.encode_column(
            input_df, 'letter', encoder)

        expected_df_shape = (3, 4)

        self.assertEqual(expected_df_shape, processed_data.shape)

    def test_get_train_and_test_sets(self):

        input_columns = ['a', 'b', 'c']
        output_column = ['d']
        all_columns = input_columns + output_column

        input_df = pd.DataFrame(np.array([[1, 1, 1, 2],
                                          [1, 1, 1, 3],
                                          [1, 1, 1, 4]]),
                                columns=all_columns)

        train_x, test_x, train_y, test_y = PipelineHelpers.get_train_and_test_sets(
            input_df, output_column, 0.33)

        expected_train_x_shape = (2, 3)
        expected_train_y_shape = (2, 1)
        expected_test_x_shape = (1, 3)
        expected_test_y_shape = (1, 1)
        self.assertEqual(expected_train_x_shape, train_x.shape)
        self.assertEqual(expected_train_y_shape, train_y.shape)
        self.assertEqual(expected_test_x_shape, test_x.shape)
        self.assertEqual(expected_test_y_shape, test_y.shape)


if __name__ == "__main__":

    unittest.main()
