import pandas as pd


class PipelineHelpers():

    def extract_data(input_data_location, columns):
        # load in the data
        raw_data_frame = pd.read_csv(
            input_data_location, encoding='unicode_escape', names=columns)
        # shuffle the data
        raw_data_frame = raw_data_frame.sample(
            frac=1).reset_index(drop=True)
        return raw_data_frame
