#!/usr/bin/env python3
"""
main.py
--------
Voert de Single Classification Tree-analyse uit op de op_spam_v1.4 dataset.
"""
from OpenFiles import Split_data
from RandomForest import RandomF
from GradBoost import GradBoost
from LogisticRegressionWithLassoPenalty import Output_Logistic
from MultinomialNaiveBayes import Output_MNB
from ClassificationTree import ClassificationTree


def main():

    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))
        
    print("Logistic regresioon with lasso penalty")
    Output_Logistic(train_df, test_df)
    print("\n")

    print("Multinomial Naive Bayes")
    Output_MNB(train_df, test_df)
    print("\n")

    print("Classification Tree")
    ClassificationTree(train_df, test_df)
    print("\n")

    print("Randomforest")
    RandomF(X_train=train_df['content'],Y_train=train_df['label'],X_test=test_df['content'],Y_test=test_df['label'],grid=False, m_features_vec=5000)
    print("\n")

    print("GradBoost")
    GradBoost(X_train=train_df['content'],Y_train=train_df['label'],X_test=test_df['content'],Y_test=test_df['label'],grid=False)

if __name__ == "__main__":
    main()
