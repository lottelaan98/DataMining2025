import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import OpenFiles
from OpenFiles import FileContent, FileLoader, Split_data
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

class TrainModel1:
    @staticmethod
    def train_model(train_df, test_df, vectorizer):
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(vectorizer, MultinomialNB())

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
    def hyperparameter_tuning1(train_df, vectorizer, vectorizer_ngram):
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(vectorizer, MultinomialNB())

        # Define the parameter grid to search
        param_grid = {
            vectorizer_ngram: [(1, 1), (1, 2)],  # Unigrams and bigrams
            'multinomialnb__alpha': [0.1, 0.5, 1.0]  # Smoothing parameter
        }

        # Set up GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)

        # Perform the grid search on the training data
        grid_search.fit(train_df['content'], train_df['label'])

        # Print the best parameters found by GridSearchCV
        print("Best Parameters:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(test_df['content'])
        accuracy = accuracy_score(test_df['label'], predictions)

        print(f"Tuned Model Accuracy: {accuracy:.4f}")
        print("Tuned Model Classification Report:")
        print(classification_report(test_df['label'], predictions))
        print("Tuned Model Confusion Matrix:")
        print(confusion_matrix(test_df['label'], predictions))

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
    
if __name__ == "__main__":
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
    print("Train:", train_df["subfolder"].value_counts().to_dict())
    print("Test :", test_df["subfolder"].value_counts().to_dict())
    print("Voorbeeld train-rij:\n", train_df.head(1))
    
    models = [
        ("modelc1", "tuned_modelc1", CountVectorizer(), "countvectorizer__ngram_range", "countvectorizer"),
        ("modelc2", "tuned_modelc2", CountVectorizer(lowercase=True, stop_words="english"), "countvectorizer__ngram_range", "countvectorizer"),
        ("modelt1", "tuned_modelt1", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfifdvectorizer"),
        ("modelt2", "tuned_modelt2", TfidfVectorizer(lowercase=True, stop_words="english"), 'tfidfvectorizer__ngram_range', "tfifdvectorizer")
    ]
    
    for i in models:
        # Train the model and then tune the model
        print("Model, ", {i[0]}, ", without hyperparameter tuning using ", {i[4]})
        model = TrainModel1.train_model(train_df, test_df, i[2])
        print("Model, ", {i[0]}, ", with hyperparameter tuning using ", {i[4]})
        best_model = TrainModel1.hyperparameter_tuning1(train_df, i[2], i[3])
        # Get feature importance
        print("Feature importance of ", {i[0]}, " using ", {i[4]})
        TrainModel1.give_feature_importance(model)
        print("Feature importance of ", {i[1]}, " using ", {i[4]})
        TrainModel1.give_feature_importance(best_model)
   
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# OUTPUT

# THE FOLLOWING IS WITH COUNTVECTORIZER, WITHOUT PREPROCESSING - STOPWORDS AND LOWERCASE
# Accuracy: 0.8938
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.91      0.88      0.89        80
#            1       0.88      0.91      0.90        80

#     accuracy                           0.89       160
#    macro avg       0.89      0.89      0.89       160
# weighted avg       0.89      0.89      0.89       160

# Confusion Matrix:
# [[70 10]
#  [ 7 73]]
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'countvectorizer__ngram_range': (1, 2), 'multinomialnb__alpha': 0.1}
# Tuned Model Accuracy: 0.8500
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.89      0.80      0.84        80
#            1       0.82      0.90      0.86        80

#     accuracy                           0.85       160
#    macro avg       0.85      0.85      0.85       160
# weighted avg       0.85      0.85      0.85       160

# Tuned Model Confusion Matrix:
# [[64 16]
#  [ 8 72]]
# Feature importance of model:

# Top 10 for class 0 (index 0):
# the                  0.061205
# and                  0.027447
# to                   0.027074
# was                  0.020083
# in                   0.015249
# we                   0.012702
# for                  0.012410
# of                   0.012264
# room                 0.011112
# it                   0.010869

# Top 10 for class 1 (index 1):
# the                  0.061631
# to                   0.030366
# and                  0.026967
# was                  0.025757
# in                   0.016458
# of                   0.013696
# room                 0.013189
# hotel                0.012339
# we                   0.012225
# it                   0.012045
# Feature importance of tuned model:

# Top 10 for class 0 (index 0):
# the                  0.032863
# and                  0.014733
# to                   0.014533
# was                  0.010778
# in                   0.008182
# we                   0.006814
# for                  0.006657
# of                   0.006578
# room                 0.005960
# it                   0.005829

# Top 10 for class 1 (index 1):
# the                  0.033110
# to                   0.016309
# and                  0.014483
# was                  0.013833
# in                   0.008836
# of                   0.007352
# room                 0.007079
# hotel                0.006623
# we                   0.006561
# it                   0.006465
# THE FOLLOWING IS WITH COUNTVECTORIZER, WITH PREPROCESSING - STOPWORDS AND LOWERCASE
# Accuracy: 0.8750
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.89      0.85      0.87        80
#            1       0.86      0.90      0.88        80

#     accuracy                           0.88       160
#    macro avg       0.88      0.88      0.87       160
# weighted avg       0.88      0.88      0.87       160

# Confusion Matrix:
# [[68 12]
#  [ 8 72]]
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'countvectorizer__ngram_range': (1, 2), 'multinomialnb__alpha': 1.0}
# Tuned Model Accuracy: 0.8812
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.92      0.84      0.88        80
#            1       0.85      0.93      0.89        80

#     accuracy                           0.88       160
#    macro avg       0.88      0.88      0.88       160
# weighted avg       0.88      0.88      0.88       160

# Tuned Model Confusion Matrix:
# [[67 13]
#  [ 6 74]]
# Feature importance of model:

# Top 10 for class 0 (index 0):
# room                 0.021683
# hotel                0.019783
# stay                 0.007723
# service              0.005983
# night                0.005223
# staff                0.005065
# chicago              0.004875
# desk                 0.004780
# did                  0.004685
# bed                  0.004147

# Top 10 for class 1 (index 1):
# room                 0.026338
# hotel                0.024641
# chicago              0.012500
# stay                 0.009432
# service              0.006527
# like                 0.005450
# desk                 0.005287
# did                  0.005124
# staff                0.004602
# night                0.004504
# Feature importance of tuned model:

# Top 10 for class 0 (index 0):
# room                 0.007255
# hotel                0.006619
# stay                 0.002584
# service              0.002002
# night                0.001747
# staff                0.001695
# chicago              0.001631
# desk                 0.001599
# did                  0.001567
# bed                  0.001387

# Top 10 for class 1 (index 1):
# room                 0.008723
# hotel                0.008161
# chicago              0.004140
# stay                 0.003124
# service              0.002162
# like                 0.001805
# desk                 0.001751
# did                  0.001697
# staff                0.001524
# night                0.001492
# THE FOLLOWING IS WITH TFIDFVECTORIZER, WITHOUT PREPROCESSING - STOPWORDS AND LOWERCASE
# Accuracy: 0.7750
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.94      0.59      0.72        80
#            1       0.70      0.96      0.81        80

#     accuracy                           0.78       160
#    macro avg       0.82      0.78      0.77       160
# weighted avg       0.82      0.78      0.77       160

# Confusion Matrix:
# [[47 33]
#  [ 3 77]]
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'multinomialnb__alpha': 0.1, 'tfidfvectorizer__ngram_range': (1, 2)}
# Tuned Model Accuracy: 0.8063
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.93      0.66      0.77        80
#            1       0.74      0.95      0.83        80

#     accuracy                           0.81       160
#    macro avg       0.83      0.81      0.80       160
# weighted avg       0.83      0.81      0.80       160

# Tuned Model Confusion Matrix:
# [[53 27]
#  [ 4 76]]
# Feature importance of model:

# Top 10 for class 0 (index 0):
# the                  0.007296
# and                  0.003406
# to                   0.003370
# was                  0.002766
# we                   0.002592
# in                   0.002112
# for                  0.001834
# of                   0.001730
# room                 0.001710
# it                   0.001693

# Top 10 for class 1 (index 1):
# the                  0.007827
# to                   0.003964
# was                  0.003654
# and                  0.003557
# we                   0.002551
# in                   0.002386
# my                   0.002126
# of                   0.002053
# room                 0.002053
# it                   0.001980
# Feature importance of tuned model:

# Top 10 for class 0 (index 0):
# the                  0.003839
# and                  0.001762
# to                   0.001749
# was                  0.001417
# we                   0.001335
# in                   0.001073
# for                  0.000925
# of                   0.000879
# room                 0.000859
# it                   0.000852

# Top 10 for class 1 (index 1):
# the                  0.004055
# to                   0.002035
# was                  0.001862
# and                  0.001814
# we                   0.001303
# in                   0.001201
# my                   0.001069
# of                   0.001036
# room                 0.001028
# it                   0.000987
# THE FOLLOWING IS WITH TFIDFVECTORIZER, WITH PREPROCESSING - STOPWORDS AND LOWERCASE
# Accuracy: 0.8063
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.92      0.68      0.78        80
#            1       0.74      0.94      0.83        80

#     accuracy                           0.81       160
#    macro avg       0.83      0.81      0.80       160
# weighted avg       0.83      0.81      0.80       160

# Confusion Matrix:
# [[54 26]
#  [ 5 75]]
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'multinomialnb__alpha': 0.1, 'tfidfvectorizer__ngram_range': (1, 2)}
# Tuned Model Accuracy: 0.8812
# Tuned Model Classification Report:
#               precision    recall  f1-score   support

#            0       0.90      0.86      0.88        80
#            1       0.87      0.90      0.88        80

#     accuracy                           0.88       160
#    macro avg       0.88      0.88      0.88       160
# weighted avg       0.88      0.88      0.88       160

# Tuned Model Confusion Matrix:
# [[69 11]
#  [ 8 72]]
# Feature importance of model:

# Top 10 for class 0 (index 0):
# room                 0.002246
# hotel                0.002044
# stay                 0.001183
# staff                0.001029
# service              0.001018
# great                0.001015
# night                0.001006
# location             0.000946
# did                  0.000945
# bed                  0.000924

# Top 10 for class 1 (index 1):
# room                 0.002803
# hotel                0.002620
# chicago              0.001888
# stay                 0.001519
# service              0.001267
# like                 0.001151
# did                  0.001069
# desk                 0.001053
# staff                0.001015
# stayed               0.000989
# Feature importance of tuned model:

# Top 10 for class 0 (index 0):
# room                 0.001416
# hotel                0.001282
# stay                 0.000720
# staff                0.000615
# service              0.000611
# great                0.000601
# night                0.000601
# did                  0.000561
# location             0.000556
# bed                  0.000547

# Top 10 for class 1 (index 1):
# room                 0.001734
# hotel                0.001624
# chicago              0.001153
# stay                 0.000908
# service              0.000747
# like                 0.000678
# did                  0.000626
# desk                 0.000616
# staff                0.000586
# stayed               0.000568
