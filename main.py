"""
Entry point for running all models on the op_spam_v1.4 dataset.

This script:
1) Loads train/test folds via Split_data (train: folds 1–4, test: fold 5),
2) Trains and evaluates:
   - Logistic Regression with L1 (LASSO) penalty
   - Multinomial Naive Bayes
   - Single Classification Tree
   - Random Forest
   - Gradient Boosting
Each model prints its own evaluation summary.
"""
from OpenFiles import Split_data
from RandomForest import RandomF
from GradBoost import GradBoost
from LogisticRegressionWithLassoPenalty import Output_Logistic
from MultinomialNaiveBayes import Output_MNB
from ClassificationTree import ClassificationTree


def main():
    # 1) Load data splits: train on folds 1–4, test on fold 5
    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))

    # 2) Logistic Regression with L1 (LASSO) penalty
    print("Logistic regresioon with lasso penalty")
    Output_Logistic(train_df, test_df)
    print("\n")

    # 3) Multinomial Naive Bayes
    print("Multinomial Naive Bayes")
    Output_MNB(train_df, test_df)
    print("\n")

    # 4) Single Classification Tree
    print("Classification Tree")
    ClassificationTree(train_df, test_df)
    print("\n")

    # 5) Random Forest (uses TF–IDF upstream in its own pipeline; grid=False means no grid search here)
    print("Randomforest")
    RandomF(X_train=train_df['content'],Y_train=train_df['label'],X_test=test_df['content'],Y_test=test_df['label'],grid=False, m_features_vec=5000)
    print("\n")

    # 6) Gradient Boosting
    print("GradBoost")
    GradBoost(X_train=train_df['content'],Y_train=train_df['label'],X_test=test_df['content'],Y_test=test_df['label'],grid=False)

if __name__ == "__main__":
    # Execute the experiment pipeline
    main()
