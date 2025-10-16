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
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=5000))

        # Train the model using the training data
        model.fit(train_df['content'], train_df['label'])

        return model

    @staticmethod
    def hyperparameter_tuning(train_df, vectorizer, vectorizer_ngram):
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=5000))

        # Define the parameter grid to search
        param_grid = {
            vectorizer_ngram: [(2,2)],  # Unigrams and bigrams
            'logisticregression__C': [0.01, 0.1, 1.0, 10.0]  # Inverse of regularization strength
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

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
            
    def accuracyvalue(model, X_test, y_test):

        y_pred = model.predict(X_test)
        
        return accuracy_score(y_test, y_pred)

def Output_Logistic(train_df, test_df):

    models = [
        ("The tfifdvectorizer model ", "The tuned tfifdvectorizer model ", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfidfvectorizer"),
    ]

    accuracy = []
    
    for i in models:
        model = TrainModel2.train_model(train_df, i[2])
        accuracy.append([i[0], TrainModel2.accuracyvalue(model, test_df["content"], test_df["label"]), model, i[4]])
        best_model = TrainModel2.hyperparameter_tuning(train_df, i[2], i[3])
        accuracy.append([i[1], TrainModel2.accuracyvalue(best_model, test_df["content"], test_df["label"]), best_model, i[4]])

    best_model = []
    highest_acc = 0

    for highest in accuracy:
        if highest[1] > highest_acc:
            highest_acc = highest[1]
            best_model = highest
    
    print("The best model is:", best_model[:2])
    print("With hyperparameters", best_model[2])
    evaluate_model(best_model[2], test_df["content"], test_df["label"])
    TrainModel2.give_feature_importance(best_model[2], best_model[3])

if __name__ == '__main__':
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    Output_Logistic(train_df, test_df)
    