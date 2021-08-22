import xgboost as xgb

from helpers import PipelineHelpers


class AbaloneClassifier():
    def __init__(self,
                 input_data_location='data/raw_data.csv',
                 batch_data_location='data/raw_data_batch_transform.csv',
                 input_columns=['sex', 'length', 'diameter', 'height', 'whole_weight',
                                'shucked_weight', 'viscera_weight', 'shell_weight'],
                 output_column=['ring']):

        self.Pipeline = []
        self.mean_squared_error = 6.0

        self.input_columns = input_columns
        self.output_column = output_column
        self.all_columns = self.input_columns + self.output_column

        self.input_data_location = input_data_location
        self.batch_data_location = batch_data_location

    def pipeline_init(self):
        print('pipeline_init ..')

        self.Pipeline.append(self.extract)
        self.Pipeline.append(self.preprocess)
        self.Pipeline.append(self.train)
        # self.Pipeline.append(self.evaluate)
        # self.Pipeline.append(self.batch_inference)

    def start(self):
        print('pipeline started')
        [step() for step in self.Pipeline]

    '''
    In this method, we
    1- Grab the raw data and construct a dataframe
    '''

    def extract(self):
        print('extraction started')
        self.raw_data_frame = PipelineHelpers.extract_data(
            self.input_data_location, self.all_columns)

    '''
    In this method, we
    1- Do the data Transform
    2- Split the raw data to train and test
    '''

    def preprocess(self):
        print('preprocessing started')

        self.raw_data_frame = PipelineHelpers.remove_outliers(
            self.raw_data_frame, 'height')
        encoder = PipelineHelpers.fit_encoder(self.raw_data_frame, 'sex')
        processed_df = PipelineHelpers.encode_column(
            self.raw_data_frame, 'sex', encoder)
        self.train_x, self.test_x, self.train_y, self.test_y = PipelineHelpers.get_train_and_test_sets(
            processed_df, self.output_column[0])

    '''
    In this method, we
    1- Train the model and store the artifacts
    '''

    def train(self):
        if self.train_x is None:
            print('Training stopped. no input data available')
            return

        print('Training started')

        # XG-Boost Params; these are passed to create the trained model
        param = {
            "max_depth": "5",
            "eta": "0.2",
            "gamma": "4",
            "min_child_weight": "6",
            "subsample": "0.7",
            "verbosity": "1",
            "objective": "reg:linear",
            "num_round": "150"
        }

        training_epochs = 350

        dtrain = xgb.DMatrix(self.train_x, label=self.train_y)
        dtest = xgb.DMatrix(self.test_x, label=self.test_y)

        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        # created a trained booster model
        self.model = xgb.train(
            param, dtrain, evals=evallist, num_boost_round=training_epochs, early_stopping_rounds=10)

    def predict(self, model, test_x, test_y=None, output_margin=False):
        if test_y is not None:
            dtest = xgb.DMatrix(test_x, label=test_y)
        else:
            dtest = xgb.DMatrix(test_x)

        return model.predict(dtest, output_margin=output_margin)

    '''
    In this method, we
    1- Use the trained model to predict the test_data and calculate the mean squared error
    '''

    def evaluate(self):
        if not self.model:
            print('Evaluation stopped. model is not trained')
            return

        print('model evaluation started')

        test_x = self.test_data[['sex', 'length', 'diameter', 'height', 'whole_weight',
                                 'shucked_weight', 'viscera_weight', 'shell_weight']]
        test_y = self.test_data[['ring']]

        predictions = self.predict(
            self.model, test_x, test_y, output_margin=True)

        # TODO: Implement calculation of the mean squared error

        print('mean squared error = {}'.format(self.mean_squared_error))

    '''
    In this method, we
    1- Batch inference/transform the data raw_data_batch_inference.csv and store the result in another CSV following this format

    original data, prediction

    e.g.
    I,0.275,0.2,0.07,0.096,0.037,0.0225,0.03,6,11.0054
    M,0.635,0.48,0.145,1.181,0.665,0.229,0.225,7.141
    '''

    def batch_inference(self):
        if not self.model:
            print('Batch inference stopped. model is not trained')
            return

        if self.mean_squared_error < 6.0:
            print('batch_inference started')


if __name__ == '__main__':
    classifier = AbaloneClassifier()
    classifier.pipeline_init()
    classifier.start()
