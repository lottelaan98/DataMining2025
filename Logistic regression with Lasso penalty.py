import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import OpenFiles
from OpenFiles import FileContent, FileLoader, Split_data
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

class TrainModel2:
    @staticmethod
    def train_model(train_df, test_df):
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(CountVectorizer(), LogisticRegression(penalty='l1', solver='saga', max_iter=1000))

        # Train the model using the training data
        model.fit(train_df['content'], train_df['label'])

        # Predict the labels for the test data
        predictions = model.predict(test_df['content'])

        # Calculate accuracy
        accuracy = accuracy_score(test_df['label'], predictions)
        print(f"Accuracy: {accuracy:.4f}")

        # Print classification report
        print("Classification Report:")
        print(classification_report(test_df['label'], predictions))

        # Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(test_df['label'], predictions))

        return model

    @staticmethod
    def hyperparameter_tuning2(train_df):
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(CountVectorizer(), LogisticRegression(penalty='l1', solver='saga', max_iter=1000))

        # Define the parameter grid to search
        param_grid = {
            'countvectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
            'logisticregression__C': [0.01, 0.1, 1.0, 10.0]  # Inverse of regularization strength
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        # Print the best parameters found by GridSearchCV
        print("Best Parameters:", grid_search.best_params_)

        return grid_search.best_estimator_
    
# But by importing OenFiles, I can still use the load_files_recursive function in another script
if __name__ == "__main__":
        # files = FileLoader.load_files_recursive(negative_polarity) 

        # print(f"Total folders loaded: {len(set(file.folder for file in files))}")
        # print(f"Subfolders in folder: {len(set(file.subfolder for file in files))}")
        # print(f"Total subfolders loaded: {len(set(file.folder for file in files)) * len(set(file.subfolder for file in files))}")
        # print(f"Total files loaded: {len(files)}")
        # print(files[0])

    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    print("Train:", train_df["subfolder"].value_counts().to_dict())
    print("Test :", test_df["subfolder"].value_counts().to_dict())
    print("Voorbeeld train-rij:\n", train_df.head(1))

    # Train the model and evaluate it
    model = TrainModel2.train_model(train_df, test_df)
    # Perform hyperparameter tuning
    best_model = TrainModel2.hyperparameter_tuning2(train_df)
    # Evaluate the best model on the test set
    predictions = best_model.predict(test_df['content'])
    accuracy = accuracy_score(test_df['label'], predictions)
    print(f"Tuned Model Accuracy: {accuracy:.4f}")
    print("Tuned Model Classification Report:")
    print(classification_report(test_df['label'], predictions))
    print("Tuned Model Confusion Matrix:")
    print(confusion_matrix(test_df['label'], predictions))
    