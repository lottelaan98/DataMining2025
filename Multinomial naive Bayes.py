import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import OpenFiles
from OpenFiles import FileContent, FileLoader, Split_data
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

class TrainModel1:
    @staticmethod
    def train_model(train_df, test_df):
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(CountVectorizer(), MultinomialNB())

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
    def hyperparameter_tuning1(train_df):
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(CountVectorizer(), MultinomialNB())

        # Define the parameter grid to search
        param_grid = {
            'countvectorizer__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
            'multinomialnb__alpha': [0.1, 0.5, 1.0]  # Smoothing parameter
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        # Print the best parameters found by GridSearchCV
        print("Best Parameters:", grid_search.best_params_)

        return grid_search.best_estimator_
    
if __name__ == "__main__":
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    print("Train:", train_df["subfolder"].value_counts().to_dict())
    print("Test :", test_df["subfolder"].value_counts().to_dict())
    print("Voorbeeld train-rij:\n", train_df.head(1))
    
    # Train the model and evaluate it
    model = TrainModel1.train_model(train_df, test_df)
    
    # Perform hyperparameter tuning
    best_model = TrainModel1.hyperparameter_tuning1(train_df)
    
    # Evaluate the best model on the test set
    predictions = best_model.predict(test_df['content'])
    accuracy = accuracy_score(test_df['label'], predictions)
    print(f"Tuned Model Accuracy: {accuracy:.4f}")
    print("Tuned Model Classification Report:")
    print(classification_report(test_df['label'], predictions))
    print("Tuned Model Confusion Matrix:")
    print(confusion_matrix(test_df['label'], predictions))

