import pandas as pd
import numpy as np


class DataInvestigation():
    def __init__(self,
                 data_location="data/raw_data.csv",
                 input_columns=['sex', 'length', 'diameter', 'height', 'whole_weight',
                                'shucked_weight', 'viscera_weight', 'shell_weight'],
                 output_column=['ring']):

        self.data_location = data_location
        self.input_columns = input_columns
        self.output_column = output_column

        self.all_columns = self.input_columns + self.output_column
        self.data = pd.read_csv(
            self.data_location, encoding='unicode_escape', names=self.all_columns)

    def describe_data(self):
        """Show stats about data, such as min and max values for each numerical column, as well as percentiles."""
        print(self.data.describe(percentiles=[0.01, .1, .25, .5, .75, .99]))

    def is_any_data_null(self):
        """Informs if any data is null and needs to be considered."""
        if self.data.isna().any().any():
            print("Data contains missing values.")
        else:
            print("No missing values in data.")

    def unique_sexes(self):
        """Informs of the number of unique sexes."""
        sexes = self.data['sex'].unique()
        print(f"Number of unique sexes: {len(sexes)}")

    def investigate_outliers(self, column):
        column_values = self.data[column].to_numpy()
        mean = column_values.mean()
        std = column_values.std()
        upper_bound = mean + (std * 3)
        outliers = [value for value in column_values if np.absolute(
            value) > upper_bound]
        print(
            f"""There were {len(outliers)} outliers in the {column} column.
            These were: {outliers}, compared to the upper bound of {upper_bound}.
            You may want to remove these.""")


if __name__ == "__main__":
    print("Training data information:")
    training_data_investigation = DataInvestigation()
    training_data_investigation.describe_data()
    training_data_investigation.is_any_data_null()
    training_data_investigation.unique_sexes()
    training_data_investigation.investigate_outliers("height")
    training_data_investigation.investigate_outliers("shell_weight")
    print("""
    ---------------------------------
    """)

    print("Test data information:")
    batch_data_investigation = DataInvestigation(
        data_location="data/raw_data_batch_transform.csv", output_column=[])
    batch_data_investigation.describe_data()
    batch_data_investigation.is_any_data_null()
    batch_data_investigation.unique_sexes()
