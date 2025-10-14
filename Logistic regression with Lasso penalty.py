import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import OpenFiles
from OpenFiles import FileContent, FileLoader, Split_data
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model

class TrainModel2:
    @staticmethod
    def train_model(train_df, test_df, vectorizer):
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=1000))

        # train_df = Word_preprocessing.preprocess_text(train_df['content'])
        # test_df = Word_preprocessing.preprocess_text(test_df['content'])

        # Train the model using the training data
        model.fit(train_df['content'], train_df['label'])

        evaluate_model(model, test_df["content"], test_df["label"])

        return model

    @staticmethod
    def hyperparameter_tuning(train_df, vectorizer, vectorizer_ngram):
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=1000))

        # Define the parameter grid to search
        param_grid = {
            vectorizer_ngram: [(1, 1), (1, 2)],  # Unigrams and bigrams
            'logisticregression__C': [0.01, 0.1, 1.0, 10.0]  # Inverse of regularization strength
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        # Print the best parameters found by GridSearchCV
        print("Best Parameters:", grid_search.best_params_)

        # Evaluate the best model on the test set
        best_model = grid_search.best_estimator_
        
        evaluate_model(best_model, test_df["content"], test_df["label"])

        return grid_search.best_estimator_
    
    def give_feature_importance(model, vectorizer):
        vec = model.named_steps[vectorizer]
        clf = model.named_steps['logisticregression']

        feature_names = vec.get_feature_names_out()

        # For binary classification: pick top positive and negative
        coefs = clf.coef_[0]                 # shape (n_features,)
        top_pos_idx = np.argsort(coefs)[-10:][::-1]   # 20 most positive
        top_neg_idx = np.argsort(coefs)[:10]          # 20 most negative

        print("Top positive words:")
        for i in top_pos_idx:
            print(f"{feature_names[i]}: {coefs[i]:.3f}")

        print("\nTop negative words:")
        for i in top_neg_idx:
            print(f"{feature_names[i]}: {coefs[i]:.3f}")

    
# But by importing OenFiles, I can still use the load_files_recursive function in another script
if __name__ == "__main__":

    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))

    models = [
        ("The countvectorizer model", "The tuned countvectorizer model", CountVectorizer(), "countvectorizer__ngram_range", "countvectorizer"),
        ("The countervectorizer model with preprocessing ", "The tuned countvectorizer model with preprocessing ", CountVectorizer(lowercase=True, stop_words="english"), "countvectorizer__ngram_range", "countvectorizer"),
        ("The tfifdvectorizer model ", "The tuned tfifdvectorizer model ", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfidfvectorizer"),
        ("The countervectorizer model with preprocessing ", "The tuned countervectorizer model with preprocessing ", TfidfVectorizer(lowercase=True, stop_words="english"), 'tfidfvectorizer__ngram_range', "tfidfvectorizer")
    ]
    
    for i in models:
        # Train the model and then tune the model
        print(i[0],", and without hyperparameter tuning")
        model = TrainModel2.train_model(train_df, test_df, i[2])
        print(i[1])
        best_model = TrainModel2.hyperparameter_tuning(train_df, i[2], i[3])
        # Get feature importance
        print("Feature importance of ", i[0])
        TrainModel2.give_feature_importance(model, i[4])
        print("Feature importance of ", i[1])
        TrainModel2.give_feature_importance(best_model, i[4])
