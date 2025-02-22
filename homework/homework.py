import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os
from glob import glob 


def outputCreation(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)

def saveModel(path, estimator):
    outputCreation("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)
    

def pregunta01():
        
    data_test = pd.read_csv("./files/input/test_data.csv.zip",index_col=False,compression="zip")
    data_train = pd.read_csv("./files/input/train_data.csv.zip",index_col = False,compression ="zip")

    def cleanse(df):
        df_copy = df.copy()
        current_year = 2021
        columns_to_drop = ['Year', 'Car_Name']
        df_copy["Age"] = current_year - df_copy["Year"]
        df_copy = df_copy.drop(columns=columns_to_drop)
        return df_copy

    data_train = cleanse(data_train)
    data_test = cleanse(data_test)
    x_train, y_train = data_train.drop(columns=["Present_Price"]), data_train["Present_Price"]
    x_test, y_test = data_test.drop(columns=["Present_Price"]), data_test["Present_Price"]

    def f_pipeline(x_train):
        categorical_features=['Fuel_Type','Selling_type','Transmission']
        numerical_features= [col for col in x_train.columns if col not in categorical_features]

        preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), categorical_features),
                    ('scaler',MinMaxScaler(),numerical_features),
                ],
            )

        pipeline=Pipeline(
                [
                    ("preprocessor",preprocessor),
                    ('feature_selection',SelectKBest(f_regression)),
                    ('classifier', LinearRegression())
                ]
            )
        return pipeline
    
    pipeline = f_pipeline(x_train)

    def hiperparams(pipeline):
        param_grid = {
        'feature_selection__k':range(1,25),
        'classifier__fit_intercept':[True,False],
        'classifier__positive':[True,False]

    }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=10,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            refit=True, 
            verbose= 1
        
        )

        return grid_search
        
    estimator = hiperparams(pipeline)
    estimator.fit(x_train, y_train)

    saveModel(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    def calc_metrics(dataset_type, y_true, y_pred):
        return {
            "type": "metrics",
            "dataset": dataset_type,
            'r2': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mad': float(median_absolute_error(y_true, y_pred)),
        }

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calc_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calc_metrics("train", y_train, y_train_pred)


    os.makedirs("files/output/", exist_ok=True)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")

if __name__ == "__main__":
    pregunta01()