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

class TrainModel2:
    @staticmethod
    def train_model(train_df, test_df, vectorizer):
        # Create a pipeline that first vectorizes the text data and then applies the Logistic Regression with Lasso penalty
        model = make_pipeline(vectorizer, LogisticRegression(penalty='l1', solver='saga', max_iter=1000))

        # train_df = Word_preprocessing.preprocess_text(train_df['content'])
        # test_df = Word_preprocessing.preprocess_text(test_df['content'])

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
    def hyperparameter_tuning2(train_df, vectorizer, vectorizer_ngram):
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
        predictions = best_model.predict(test_df['content'])
        accuracy = accuracy_score(test_df['label'], predictions)

        print(f"Tuned Model Accuracy: {accuracy:.4f}")
        print("Tuned Model Classification Report:")
        print(classification_report(test_df['label'], predictions))
        print("Tuned Model Confusion Matrix:")
        print(confusion_matrix(test_df['label'], predictions))

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
    # print("Train:", train_df["subfolder"].value_counts().to_dict())
    # print("Test :", test_df["subfolder"].value_counts().to_dict())
    # print("Voorbeeld train-rij:\n", train_df.head(1))

    # Train the model and then tune the model and evaluate it without preprocessing of stopwords and lowercase
    print("THE FOLLOWING IS WITH COUNTVECTORIZER, WITHOUT PREPROCESSING - STOPWORDS AND LOWERCASE")
    modelc1 = TrainModel2.train_model(train_df, test_df, CountVectorizer())
    best_modelc1 = TrainModel2.hyperparameter_tuning2(train_df, CountVectorizer(), 'countvectorizer__ngram_range')
    # Get feature importance
    print("Feature importance of model:")
    TrainModel2.give_feature_importance(modelc1, 'countvectorizer')
    print("Feature importance of tuned model:")
    TrainModel2.give_feature_importance(best_modelc1, 'countvectorizer')

    # Train the model and then tune, with preprocessing of stopwords and lowercasing
    print("THE FOLLOWING IS WITH COUNTVECTORIZER, WITH PREPROCESSING - STOPWORDS AND LOWERCASE")
    modelc2 = TrainModel2.train_model(train_df, test_df, CountVectorizer(lowercase=True, stop_words="english"))
    best_modelc2 = TrainModel2.hyperparameter_tuning2(train_df, CountVectorizer(lowercase=True, stop_words="english"), 'countvectorizer__ngram_range')
    # Get feature importance
    print("Feature importance of model:")
    TrainModel2.give_feature_importance(modelc2, 'countvectorizer')
    print("Feature importance of tuned model:")
    TrainModel2.give_feature_importance(best_modelc2, 'countvectorizer')

    # Train the model and then tune, with preprocessing of stopwords and lowercasing
    print("THE FOLLOWING IS WITH TFIDFVECTORIZER, WITHOUT PREPROCESSING - STOPWORDS AND LOWERCASE")
    modelt1 = TrainModel2.train_model(train_df, test_df, TfidfVectorizer())
    best_modelt1 = TrainModel2.hyperparameter_tuning2(train_df, TfidfVectorizer(), 'tfidfvectorizer__ngram_range')
    # Get feature importance
    print("Feature importance of model:")
    TrainModel2.give_feature_importance(modelt1, 'tfidfvectorizer')
    print("Feature importance of tuned model:")
    TrainModel2.give_feature_importance(best_modelt1, 'tfidfvectorizer')

    # Train the model and then tune, with preprocessing of stopwords and lowercasing
    print("THE FOLLOWING IS WITH TFIDFVECTORIZER, WITH PREPROCESSING - STOPWORDS AND LOWERCASE")
    modelt2 = TrainModel2.train_model(train_df, test_df, TfidfVectorizer(lowercase=True, stop_words="english"))
    best_modelt2 = TrainModel2.hyperparameter_tuning2(train_df, TfidfVectorizer(lowercase=True, stop_words="english"), 'tfidfvectorizer__ngram_range')
    # Get feature importance
    print("Feature importance of model:")
    TrainModel2.give_feature_importance(modelt2, 'tfidfvectorizer')
    print("Feature importance of tuned model:")
    TrainModel2.give_feature_importance(best_modelt2, 'tfidfvectorizer')
   


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# THE FOLLOWING IS WITH COUNTVECTORIZER, WITHOUT PREPROCESSING - STOPWORDS AND LOWERCASE
# Accuracy: 0.8562
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.84      0.88      0.86        80
#            1       0.87      0.84      0.85        80
#     accuracy                           0.86       160
#    macro avg       0.86      0.86      0.86       160
# weighted avg       0.86      0.86      0.86       160

# Confusion Matrix:
# [[70 10]
#  [13 67]]

# Best Parameters: {'countvectorizer__ngram_range': (1, 2), 'logisticregression__C': 10.0}
# Tuned Model Accuracy: 0.8625
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.84      0.90      0.87        80
#            1       0.89      0.82      0.86        80

#     accuracy                           0.86       160
#    macro avg       0.86      0.86      0.86       160
# weighted avg       0.86      0.86      0.86       160

# Tuned Model Confusion Matrix:
# [[72  8]
#  [14 66]]

# Feature importance of model:
# Top positive words:
# chicago: 1.050
# finally: 0.848
# recently: 0.810
# luxury: 0.724
# decided: 0.702
# make: 0.688
# be: 0.574
# prices: 0.545
# how: 0.543
# smelled: 0.537

# Top negative words:
# star: -0.827
# conference: -0.660
# times: -0.622
# great: -0.616
# can: -0.582
# floor: -0.567
# location: -0.550
# open: -0.540
# tiny: -0.522
# coffee: -0.517

# Feature importance of tuned model:
# Top positive words:
# chicago: 1.134
# be: 0.533
# finally: 0.466
# make: 0.454
# at the: 0.414
# recently: 0.404
# luxury: 0.393
# hotel: 0.378
# seemed: 0.374
# and the: 0.372

# Top negative words:
# very: -0.587
# great: -0.541
# location: -0.501
# conference: -0.469
# star: -0.461
# have: -0.431
# floor: -0.411
# can: -0.399
# this: -0.398
# bed: -0.395

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# THE FOLLOWING IS WITH COUNTVECTORIZER, WITH PREPROCESSING - STOPWORDS AND LOWERCASE

# Accuracy: 0.8375
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.84      0.84      0.84        80
#            1       0.84      0.84      0.84        80

#     accuracy                           0.84       160
#    macro avg       0.84      0.84      0.84       160
# weighted avg       0.84      0.84      0.84       160

# Confusion Matrix:
# [[67 13]
#  [13 67]]

# Best Parameters: {'countvectorizer__ngram_range': (1, 1), 'logisticregression__C': 10.0}
# Tuned Model Accuracy: 0.8187
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.82      0.81      0.82        80
#            1       0.81      0.82      0.82        80

#     accuracy                           0.82       160
#    macro avg       0.82      0.82      0.82       160
# weighted avg       0.82      0.82      0.82       160

# Tuned Model Confusion Matrix:
# [[65 15]
#  [14 66]]

# Feature importance of model:
# Top positive words:
# recently: 1.420
# millennium: 1.398
# decided: 1.376
# turned: 1.374
# chicago: 1.250
# finally: 1.211
# cleaned: 1.175
# pictures: 1.167
# luxury: 1.159
# prices: 1.075

# Top negative words:
# star: -1.803
# returned: -1.285
# concierge: -1.263
# michigan: -1.035
# tell: -1.020
# security: -1.019
# coffee: -0.994
# tiny: -0.991
# walk: -0.977
# phone: -0.966

# Feature importance of tuned model:
# Top positive words:
# chicago: 1.731
# recently: 1.702
# luxury: 1.393
# finally: 1.367
# decided: 1.346
# smelled: 1.272
# cleaned: 1.259
# millennium: 1.222
# make: 1.106
# terrible: 1.061

# Top negative words:
# star: -2.032
# concierge: -1.464
# street: -1.427
# location: -1.380
# walk: -1.334
# phone: -1.289
# tell: -1.265
# conference: -1.238
# construction: -1.187
# floor: -1.172

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# THE FOLLOWING IS WITH TFIDFVECTORIZER, WITHOUT PREPROCESSING - STOPWORDS AND LOWERCASE

# Accuracy: 0.7750
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.74      0.84      0.79        80
#            1       0.81      0.71      0.76        80

#     accuracy                           0.78       160
#    macro avg       0.78      0.78      0.77       160
# weighted avg       0.78      0.78      0.77       160

# Confusion Matrix:
# [[67 13]
#  [23 57]]

# Best Parameters: {'logisticregression__C': 10.0, 'tfidfvectorizer__ngram_range': (1, 1)}
# Tuned Model Accuracy: 0.8375
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.82      0.86      0.84        80
#            1       0.86      0.81      0.83        80

#     accuracy                           0.84       160
#    macro avg       0.84      0.84      0.84       160
# weighted avg       0.84      0.84      0.84       160

# Tuned Model Confusion Matrix:
# [[69 11]
#  [15 65]]

# Feature importance of model:
# Top positive words:
# chicago: 19.996
# my: 3.993
# luxury: 3.958
# was: 3.899
# smell: 3.292
# be: 2.804
# hotel: 2.267
# finally: 2.012
# recently: 2.010
# when: 1.848

# Top negative words:
# great: -2.537
# location: -2.059
# conference: -1.330
# star: -1.329
# is: -1.009
# very: -0.798
# no: -0.782
# floor: -0.769
# years: 0.000
# 000: 0.000

# Feature importance of tuned model:
# Top positive words:
# chicago: 32.610
# decided: 18.171
# luxury: 16.747
# prices: 16.365
# recently: 14.710
# finally: 14.639
# make: 13.611
# be: 13.122
# turned: 11.593
# arrive: 11.466

# Top negative words:
# star: -15.401
# conference: -9.976
# rate: -9.299
# try: -9.254
# returned: -8.916
# construction: -8.406
# floor: -8.271
# world: -8.211
# properties: -7.965
# can: -7.725

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# THE FOLLOWING IS WITH TFIDFVECTORIZER, WITH PREPROCESSING - STOPWORDS AND LOWERCASE
# Accuracy: 0.7562
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.72      0.84      0.77        80
#            1       0.81      0.68      0.73        80

#     accuracy                           0.76       160
#    macro avg       0.76      0.76      0.75       160
# weighted avg       0.76      0.76      0.75       160

# Confusion Matrix:
# [[67 13]
#  [26 54]]

# Best Parameters: {'logisticregression__C': 10.0, 'tfidfvectorizer__ngram_range': (1, 1)}
# Tuned Model Accuracy: 0.8313
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.83      0.84      0.83        80
#            1       0.84      0.82      0.83        80

#     accuracy                           0.83       160
#    macro avg       0.83      0.83      0.83       160
# weighted avg       0.83      0.83      0.83       160

# Tuned Model Confusion Matrix:
# [[67 13]
#  [14 66]]

# Feature importance of model:
# Top positive words:
# chicago: 17.884
# luxury: 5.408
# finally: 5.017
# smell: 4.474
# recently: 3.414
# decided: 3.052
# smelled: 2.780
# experience: 2.278
# hotel: 2.213
# like: 1.904

# Top negative words:
# location: -3.344
# great: -2.946
# star: -2.661
# elevators: -1.895
# conference: -1.572
# floor: -1.170
# construction: -0.372
# zone: 0.000
# 000: 0.000
# 00a: 0.000

# Feature importance of tuned model:
# Top positive words:
# chicago: 33.149
# decided: 14.549
# recently: 14.476
# millennium: 14.441
# turned: 14.283
# smelled: 13.546
# make: 12.965
# prices: 12.306
# finally: 12.123
# website: 11.568

# Top negative words:
# star: -16.217
# construction: -12.580
# returned: -12.564
# location: -11.073
# outdated: -10.638
# properties: -10.231
# cool: -10.019
# security: -9.755
# elevators: -9.629
# phone: -9.461