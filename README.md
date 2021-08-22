# Abalone age classifier

This project contains a pipeline and data to train and save an XGBoost model that predicts the age of abalone from physical measurements.
The age of abalone is determined by cutting the shell through the cone and counting the number of rings through a microscope.

## Setup

This project requires Python 3.

You must install the project requirements in order to run the pipeline:

```
pip install -r requirements
```

It is recommended to do so in a virtual environment using Python 3.7+.

## Running the pipeline

The pipeline can be run simply by running:

```
python entry_point.py
```

This will initiate the pipeline and execute all of its steps, which are as follows:

1. `extract`: load in the raw data from the input csv data file.
2. `preprocess`: remove extreme outliers, create one-hot encodings of categorical columns, and split the data into training and test sets.
3. `train`: create and train an XGBoost model before saving it to the `models` folder.
4. `evaluate`: get age/ring predictions for abalones in the test set and evaluate the model by calculating the mean squared error.
5. `batch_inference`: load in data and get age predictions before saving the full data in a csv file in `data`.

## Tests

The tests can be run as follows:

```
python test.py
```

## My approach

1. Run investigations on the data to get a better idea of how it looked. When doing this I discovered there was potentially some incorrect inputs in the `height` column (max value of 1.13 vs 75th percentile of 0.165) so I decided I would remove extreme outliers in the `height` column in the preprocessing step of the pipeline.
2. Implement each step in the pipeline chronologically, adding tests and committing as I finished each step. At the start of this I decided to add a `PipelineHelpers` class in a new file which contained functions to use in the pipeline in order write easy to test functions.
3. Run the full pipeline to generate the model and predictions.
4. Any necessary cleanup.

### Future improvements

There are a few changes I would make if not for time limitations:

- Implement a grid search to find the best parameters for the model.
- Generalise the loading/transforming of the training and batch data, as there is some reused code between the `preprocess` and `batch_inference` methods.
- Generalise the pipeline more so that it can be used in more custom ways.
  For example, to only run predictions and not training, or save a new model with a new version rather than simply overwriting the model.
