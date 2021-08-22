import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class PipelineHelpers():

    def extract_data(input_data_location: str, columns: list):
        # load in the data
        raw_data_frame = pd.read_csv(
            input_data_location, encoding='unicode_escape', names=columns)
        # shuffle the data
        raw_data_frame = raw_data_frame.sample(
            frac=1).reset_index(drop=True)
        return raw_data_frame

    def remove_outliers(df: pd.DataFrame, column: str):
        column_values = df[column].to_numpy()
        mean = column_values.mean()
        std = column_values.std()
        upper_bound = mean + (std * 3)

        df['contains_outlier'] = np.absolute(df[column]) > upper_bound
        not_outliers = df[df['contains_outlier'] == False]
        return not_outliers.drop('contains_outlier', axis=1)

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
        processed_data = pd.concat(
            [encoded_labels_df,
             df.drop(column, axis=1)],
            axis=1
        )

        return processed_data
