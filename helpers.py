from math import exp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class PipelineHelpers():

    def extract_data(input_data_location: str, columns: list):
        # load in the data
        raw_data_frame = pd.read_csv(
            input_data_location, encoding='unicode_escape', names=columns).dropna()
        # shuffle the data
        raw_data_frame = raw_data_frame.sample(
            frac=1).reset_index(drop=True)
        return raw_data_frame

    def remove_outliers(df: pd.DataFrame, column: str):
        column_values = df[column].to_numpy()
        mean = column_values.mean()
        std = column_values.std()
        upper_bound = mean + (std * 3)

        contains_outlier_column = 'contains_outlier'
        df[contains_outlier_column] = np.absolute(df[column]) > upper_bound
        not_outliers = df[df[contains_outlier_column] == False]
        test = not_outliers.drop(contains_outlier_column, axis=1)
        return test

    def fit_encoder(df: pd.DataFrame, column: str):
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(df[[column]])
        return ohe

    def encode_column(df: pd.DataFrame, column: str, ohe: OneHotEncoder):
        # encode labels
        encoded_labels = ohe.transform(df[[column]])
        encoded_feature_names = ohe.get_feature_names()
        encoded_labels_df = pd.DataFrame(
            encoded_labels, columns=encoded_feature_names)

        # transform df to include encoded data
        encoded_labels_df.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        processed_data = pd.concat(
            [encoded_labels_df,
             df.drop(column, axis=1)],
            axis=1
        )

        return processed_data

    def get_train_and_test_sets(df: pd.DataFrame, output_column: list, test_size=0.2):
        input_data = df.drop(output_column, axis=1)
        output_data = df[output_column]

        train_x, test_x, train_y, test_y = train_test_split(
            input_data, output_data, test_size=test_size)

        return train_x, test_x, train_y, test_y

    def mean_squared_error(actual: np.array, expected: np.array):
        predicted_diff = actual - expected
        predicted_diff_squared = predicted_diff * predicted_diff
        return predicted_diff_squared.sum() / predicted_diff.size
