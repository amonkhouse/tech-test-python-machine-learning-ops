import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np


class AbaloneClassifier():
    def __init__(self):
        self.raw_data_frame = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.Pipeline = []
        self.mean_squared_error = 6.0

    def pipeline_init(self):
        print('pipeline_init ..')

        self.Pipeline.append(self.extract)
        self.Pipeline.append(self.preprocess)
        self.Pipeline.append(self.train)
        self.Pipeline.append(self.evaluate)
        self.Pipeline.append(self.batch_inference)

    def start(self):
        print('pipeline srarted')
        [step() for step in self.Pipeline]

    '''
    In this method, we
    1- Grab the raw data and construct a dataframe
    '''

    def extract(self):
        print('extraction started')

        header = ['sex', 'length', 'diameter', 'height', 'whole_weight',
                  'shucked_weight', 'viscera_weight', 'shell_weight', 'ring']
        self.raw_data_frame = pd.read_csv(
            'data/raw_data.csv', encoding='unicode_escape', names=header)

        # shuffling the data
        self.raw_data_frame = self.raw_data_frame .sample(
            frac=1).reset_index(drop=True)

    '''
    In this method, we
    1- Do the data Transform
    2- Split the raw data to train and and test
    '''

    def preprocess(self):
        print('preprocessing started')

    '''
    In this method, we
    1- Train the model and store the artifacts
    '''

    def train(self):
        if not self.train_data:
            print('Training stopped. no input data available')
            return

        print('training started')

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

        train_x = self.train_data[['sex', 'length', 'diameter', 'height', 'whole_weight',
                                  'shucked_weight', 'viscera_weight', 'shell_weight']]
        train_y = self.train_data[['ring']]

        test_x = self.test_data[['sex', 'length', 'diameter', 'height', 'whole_weight',
                                 'shucked_weight', 'viscera_weight', 'shell_weight']]
        test_y = self.test_data[['ring']]

        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)

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
