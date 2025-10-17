"""
This module trains and evaluates a Multinomial Naive Bayes text classifier.
It builds a scikit-learn pipeline with vectorization (TF–IDF), performs
hyperparameter tuning via 5-fold cross-validation (GridSearchCV) over
alpha (smoothing) and n-gram range, reports performance, and prints the
most indicative tokens per class using the learned log probabilities.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model
from OpenFiles import Split_data

class TrainModel1:
    @staticmethod
    def train_model(train_df, vectorizer):
        # Pipeline: vectorize text -> Multinomial Naive Bayes (suitable for count/TF–IDF features)
        model = make_pipeline(vectorizer, MultinomialNB())

        # Fit on training data
        model.fit(train_df['content'], train_df['label'])
        
        return model

    @staticmethod
    def hyperparameter_tuning1(train_df, vectorizer, vectorizer_ngram):
        # Same pipeline, used as estimator for GridSearchCV
        model = make_pipeline(vectorizer, MultinomialNB())

        # Parameter grid:
        # - ngram_range of the vectorizer (compare unigrams vs. unigrams+bigrams)
        # - alpha (additive smoothing) of MultinomialNB
        param_grid = {
            vectorizer_ngram: [(1, 2)],  # Unigrams and bigrams
            'multinomialnb__alpha': [0.1, 0.5, 1.0]  # Smoothing parameter
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        return grid_search.best_estimator_
    
    def give_feature_importance(pipeline_or_grid):
        # Accept either a fitted Pipeline or a fitted GridSearchCV
        pipe = pipeline_or_grid.best_estimator_ if isinstance(pipeline_or_grid, GridSearchCV) else pipeline_or_grid
        
        # Get the vectorizer and the Naive Bayes classifier from the pipeline
        steps = pipe.named_steps
        # find vectorizer
        vec = next(v for v in steps.values() if isinstance(v, (CountVectorizer, TfidfVectorizer)))
        # find multinomial NB
        nb  = next(v for v in steps.values() if isinstance(v, MultinomialNB))

        # Feature names (tokens) and per-class log-probabilities
        feature_names = vec.get_feature_names_out()

        # Print top 10 tokens per class by highest conditional probability P(token|class)
        for i, cls in enumerate(nb.classes_):
            logp = nb.feature_log_prob_[i]
            idx = np.argsort(logp)[-10:][::-1]
            feats = feature_names[idx]
            probs = np.exp(logp[idx])
            print(f"\nTop 10 for class {cls} (index {i}):")
            for f, p in zip(feats, probs):
                print(f"{f:20s} {p:.6f}")

    def accuracyvalue(model, X_test, y_test):
        # Simple accuracy on the test set
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
def Output_MNB(train_df, test_df):
    # Variants to evaluate: (name, tuned name, vectorizer, param key for grid, pipeline step name)
    models = [
        ("The tfifdvectorizer model ", "The tuned tfifdvectorizer model ", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfifdvectorizer")]

    accuracy = []

    # Train baseline and tuned model; store accuracy and model object
    for i in models:
        model = TrainModel1.train_model(train_df, i[2])
        accuracy.append([i[0], TrainModel1.accuracyvalue(model, test_df["content"], test_df["label"]), model])
        best_model = TrainModel1.hyperparameter_tuning1(train_df, i[2], i[3])
        accuracy.append([i[1], TrainModel1.accuracyvalue(best_model, test_df["content"], test_df["label"]), best_model])
        
    # Select best model by accuracy
    best_model = []
    highest_acc = 0
    for highest in accuracy:
        if highest[1] > highest_acc:
            highest_acc = highest[1]
            best_model = highest
    
    print("The best model is:", best_model[:2])
    print("With hyperparameters", best_model[2])
    
    # Detailed evaluation (e.g., classification report / confusion matrix inside evaluate_model)
    evaluate_model(best_model[2], test_df["content"], test_df["label"])
    
    # Show top tokens per class
    TrainModel1.give_feature_importance(best_model[2])
   
if __name__ == '__main__':
    # Train on folds 1–4 and test on fold 5 (consistent with 5-fold CV usage)
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    Output_MNB(train_df, test_df)
    

