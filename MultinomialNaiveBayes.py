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
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(vectorizer, MultinomialNB())

        # Train the model using the training data
        model.fit(train_df['content'], train_df['label'])
        
        return model

    @staticmethod
    def hyperparameter_tuning1(train_df, vectorizer, vectorizer_ngram):
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(vectorizer, MultinomialNB())

        # Define the parameter grid to search
        param_grid = {
            vectorizer_ngram: [(2, 2)],  # Unigrams and bigrams
            'multinomialnb__alpha': [0.1, 0.5, 1.0]  # Smoothing parameter
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        return grid_search.best_estimator_
    
    def give_feature_importance(pipeline_or_grid):
        pipe = pipeline_or_grid.best_estimator_ if isinstance(pipeline_or_grid, GridSearchCV) else pipeline_or_grid
        
        # Grab steps
        steps = pipe.named_steps
        # find vectorizer
        vec = next(v for v in steps.values() if isinstance(v, (CountVectorizer, TfidfVectorizer)))
        # find multinomial NB
        nb  = next(v for v in steps.values() if isinstance(v, MultinomialNB))

        feature_names = vec.get_feature_names_out()

        for i, cls in enumerate(nb.classes_):
            logp = nb.feature_log_prob_[i]
            idx = np.argsort(logp)[-10:][::-1]
            feats = feature_names[idx]
            probs = np.exp(logp[idx])
            print(f"\nTop 10 for class {cls} (index {i}):")
            for f, p in zip(feats, probs):
                print(f"{f:20s} {p:.6f}")

    def accuracyvalue(model, X_test, y_test):
       
        y_pred = model.predict(X_test)
        
        return accuracy_score(y_test, y_pred)
    
def Output_MNB(train_df, test_df):

    models = [
        ("The tfifdvectorizer model ", "The tuned tfifdvectorizer model ", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfifdvectorizer")]

    accuracy = []
    
    for i in models:
        model = TrainModel1.train_model(train_df, i[2])
        accuracy.append([i[0], TrainModel1.accuracyvalue(model, test_df["content"], test_df["label"]), model])
        best_model = TrainModel1.hyperparameter_tuning1(train_df, i[2], i[3])
        accuracy.append([i[1], TrainModel1.accuracyvalue(best_model, test_df["content"], test_df["label"]), best_model])

    best_model = []
    highest_acc = 0

    for highest in accuracy:
        if highest[1] > highest_acc:
            highest_acc = highest[1]
            best_model = highest
    
    print("The best model is:", best_model[:2])
    print("With hyperparameters", best_model[2])
    evaluate_model(best_model[2], test_df["content"], test_df["label"])
    TrainModel1.give_feature_importance(best_model[2])
   
if __name__ == '__main__':
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    Output_MNB(train_df, test_df)
    

