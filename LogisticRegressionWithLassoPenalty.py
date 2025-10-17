"""
This module trains and evaluates a Logistic Regression classifier with L1 regularization (LASSO) for text data. It builds a scikit-learn pipeline with vectorization (TF–IDF), 
performs hyperparameter tuning using 5-fold cross-validation (GridSearchCV) on C and n-gram range, reports performance, and shows the most important words (feature weights from coefficients).
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model
from OpenFiles import Split_data

class TrainModel2:
    @staticmethod
    def train_model(train_df, vectorizer):
         # Pipeline: vectorize text -> Logistic Regression with L1 (LASSO) and saga solver (efficient for sparse TF-IDF)
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=5000))

        # Train the pipeline on the training set
        model.fit(train_df['content'], train_df['label'])

        return model

    @staticmethod
    def hyperparameter_tuning(train_df, vectorizer, vectorizer_ngram):
        # Same pipeline as above, used as estimator for GridSearchCV
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=5000))

        # Parameter grid:
        # - ngram_range of the vectorizer (unigrams and bigrams)
        # - C (inverse regularization strength): small C = stronger regularization; large C = weaker
        param_grid = {
            vectorizer_ngram: [(1, 2)],  # Unigrams and bigrams
            'logisticregression__C': [0.01, 0.1, 1.0, 10.0]  # Inverse of regularization strength
        }

        # 5-fold cross-validation, parallelized with n_jobs=-1
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        return grid_search.best_estimator_
    
    def give_feature_importance(model, vectorizer):
        # Extract vectorizer and classifier from the pipeline
        vec = model.named_steps[vectorizer]
        clf = model.named_steps['logisticregression']

        # Get feature names (tokens) from the vectorizer
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
            
    def accuracyvalue(model, X_test, y_test):
         # Simple accuracy score on the test set
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

def Output_Logistic(train_df, test_df):
    # Define models to test: (name, tuned_name, vectorizer, param_grid key, pipeline step name)
    models = [
        ("The tfifdvectorizer model ", "The tuned tfifdvectorizer model ", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfidfvectorizer"),
    ]

    accuracy = []

    # Train baseline and tuned variant; store accuracy, model, and step name
    for i in models:
        model = TrainModel2.train_model(train_df, i[2])
        accuracy.append([i[0], TrainModel2.accuracyvalue(model, test_df["content"], test_df["label"]), model, i[4]])
        best_model = TrainModel2.hyperparameter_tuning(train_df, i[2], i[3])
        accuracy.append([i[1], TrainModel2.accuracyvalue(best_model, test_df["content"], test_df["label"]), best_model, i[4]])

    # Select the best model based on accuracy
    best_model = []
    highest_acc = 0
    for highest in accuracy:
        if highest[1] > highest_acc:
            highest_acc = highest[1]
            best_model = highest
    
    print("The best model is:", best_model[:2])
    print("With hyperparameters", best_model[2])

    # Detailed evaluation (confusion matrix, classification report, etc.)
    evaluate_model(best_model[2], test_df["content"], test_df["label"])

    # Show most informative terms
    TrainModel2.give_feature_importance(best_model[2], best_model[3])

if __name__ == '__main__':
    # Train on folds 1–4 and test on fold 5 (as in cross-validation setup)
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    Output_Logistic(train_df, test_df)
    
