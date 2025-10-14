import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import OpenFiles
from OpenFiles import FileContent, FileLoader, Split_data, Word_preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model

class TrainModel1:
    @staticmethod
    def train_model(train_df, test_df, vectorizer):
        # Create a pipeline that first vectorizes the text data and then applies the Multinomial Naive Bayes classifier
        model = make_pipeline(vectorizer, MultinomialNB())

        # Train the model using the training data
        model.fit(train_df['content'], train_df['label'])

        evaluate_model(model, test_df["content"], test_df["label"])

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

        evaluate_model(best_model, test_df["content"], test_df["label"])

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

    models = [
        ("The countvectorizer model", "The tuned countvectorizer model", CountVectorizer(), "countvectorizer__ngram_range", "countvectorizer"),
        ("The countervectorizer model with preprocessing ", "The tuned countvectorizer model with preprocessing ", CountVectorizer(lowercase=True, stop_words="english"), "countvectorizer__ngram_range", "countvectorizer"),
        ("The tfifdvectorizer model ", "The tuned tfifdvectorizer model ", TfidfVectorizer(), 'tfidfvectorizer__ngram_range', "tfifdvectorizer"),
        ("The countervectorizer model with preprocessing ", "The tuned countervectorizer model with preprocessing ", TfidfVectorizer(lowercase=True, stop_words="english"), 'tfidfvectorizer__ngram_range', "tfifdvectorizer")
    ]
    
    for i in models:
        # Train the model and then tune the model
        print(i[0],", and without hyperparameter tuning")
        model = TrainModel1.train_model(train_df, test_df, i[2])
        print(i[1])
        best_model = TrainModel1.hyperparameter_tuning1(train_df, i[2], i[3])
        # Get feature importance
        print("Feature importance of ", i[0])
        TrainModel1.give_feature_importance(model)
        print("Feature importance of ", i[1])
        TrainModel1.give_feature_importance(best_model)
   
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# OUTPUT

# The countvectorizer model , and without hyperparameter tuning
# accuracy:  0.89375
# precision:  0.8795180722891566
# recall:  0.9125
# f1:  0.8957055214723927
# confusion matrix:  
# [[70 10]
#  [ 7 73]]
# The tuned countvectorizer model
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'countvectorizer__ngram_range': (1, 2), 'multinomialnb__alpha': 0.1}
# accuracy:  0.85
# precision:  0.8181818181818182
# recall:  0.9
# f1:  0.8571428571428571
# confusion matrix:  [[64 16]
#  [ 8 72]]
# Feature importance of  The countvectorizer model

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
# Feature importance of  The tuned countvectorizer model

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
# The countervectorizer model with preprocessing  , and without hyperparameter tuning
# accuracy:  0.875
# precision:  0.8571428571428571
# recall:  0.9
# f1:  0.8780487804878049
# confusion matrix:  [[68 12]
#  [ 8 72]]
# The tuned countvectorizer model with preprocessing
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'countvectorizer__ngram_range': (1, 2), 'multinomialnb__alpha': 1.0}
# accuracy:  0.88125
# precision:  0.8505747126436781
# recall:  0.925
# f1:  0.8862275449101796
# confusion matrix:  [[67 13]
#  [ 6 74]]
# Feature importance of  The countervectorizer model with preprocessing

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
# Feature importance of  The tuned countvectorizer model with preprocessing

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
# The tfifdvectorizer model  , and without hyperparameter tuning
# accuracy:  0.775
# precision:  0.7
# recall:  0.9625
# f1:  0.8105263157894737
# confusion matrix:  [[47 33]
#  [ 3 77]]
# The tuned tfifdvectorizer model
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'multinomialnb__alpha': 0.1, 'tfidfvectorizer__ngram_range': (1, 2)}
# accuracy:  0.80625
# precision:  0.7378640776699029
# recall:  0.95
# f1:  0.8306010928961749
# confusion matrix:  [[53 27]
#  [ 4 76]]
# Feature importance of  The tfifdvectorizer model 

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
# Feature importance of  The tuned tfifdvectorizer model

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
# The countervectorizer model with preprocessing  , and without hyperparameter tuning
# accuracy:  0.80625
# precision:  0.7425742574257426
# recall:  0.9375
# f1:  0.8287292817679558
# confusion matrix:  [[54 26]
#  [ 5 75]]
# The tuned countervectorizer model with preprocessing
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# Best Parameters: {'multinomialnb__alpha': 0.1, 'tfidfvectorizer__ngram_range': (1, 2)}
# accuracy:  0.88125
# precision:  0.8674698795180723
# recall:  0.9
# f1:  0.8834355828220859
# confusion matrix:  [[69 11]
#  [ 8 72]]
# Feature importance of  The countervectorizer model with preprocessing 

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
# Feature importance of  The tuned countervectorizer model with preprocessing

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