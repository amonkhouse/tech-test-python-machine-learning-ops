import unittest
import pandas as pd
from unittest.case import TestCase
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

    def test_encode_text_column(self):
        input_df = pd.DataFrame({'letter': ['a', 'b', 'c'],
                                 'count': [1, 2, 3]})

        encoder = PipelineHelpers.fit_encoder(input_df, 'letter')
        processed_data = PipelineHelpers.encode_column(
            input_df, 'letter', encoder)

        expected_df_shape = (3, 4)

        self.assertEquals(expected_df_shape, processed_data.shape)


if __name__ == "__main__":

    unittest.main()
