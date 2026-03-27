# Weather Prediction ML Model

This project implements a machine learning model to predict weather conditions (rain, sunny, or snow) based on historical weather data.

## Project Overview
The goal of this project is to build and evaluate a classification model that can predict the `precip_type` (precipitation type) given various atmospheric conditions.

## Dataset
The dataset used for this project is `weatherHistory.csv`.

### Data Cleaning and Preprocessing
1.  **Missing Values**: Checked for null values and filled missing `precip_type` entries with 'sunny'.
2.  **Column Renaming**: Column names were standardized by converting them to lowercase and replacing spaces with underscores.
3.  **Data Resampling**: To address class imbalance, the dataset was resampled to have an equal number of 'rain', 'snow', and 'sunny' observations.

## Features Used
The following features were used for training the model:
*   `temperature_(c)`
*   `humidity`
*   `wind_speed_(km/h)`
*   `visibility_(km)`
*   `pressure_(millibars)`

## Target Variable
The target variable `precip_type` was mapped to numerical values:
*   `rain`: 0
*   `sunny`: 1
*   `snow`: 2

## Model Training
1.  **Preprocessing Pipeline**: A `MinMaxScaler` was applied to the features to scale them.
2.  **Model**: A `RandomForestClassifier` was chosen for the classification task.
3.  **Hyperparameter Tuning**: `RandomizedSearchCV` with `KFold` cross-validation was used to find the best hyperparameters for the `RandomForestClassifier`.
    *   `model__max_depth`: [3, 5, 7]
    *   `model__n_estimators`: [100, 200, 300]
    *   `model__criterion`: ['gini', 'entropy']
4.  **Training**: The model was trained on 80% of the data, with 20% reserved for testing.

## Model Evaluation
*   **Cross-validation Score**: The cross-validation accuracy on the training set was consistently high.
*   **Classification Report**: A classification report was generated to evaluate precision, recall, and f1-score on the test set.

## Prediction Example
The trained model can predict the `precip_type` for new, unseen data. An example prediction was made with `sample_data=[[11.183333,0.96,10.4811,4.025,994.63]]`, which resulted in a 'sunny' prediction.

## Model Persistence
The trained model was saved using `joblib` as `Weather_prediction.joblib` for future use.
