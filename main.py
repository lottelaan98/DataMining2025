#!/usr/bin/env python3
"""
main.py
--------
Voert de Single Classification Tree-analyse uit op de op_spam_v1.4 dataset.
"""
from OpenFiles import Split_data
from RandomForest import train_single_tree, get_top_features, RandomF
from GradBoost import GradBoost
from evaluate import evaluate_model


def main():
    # === 1. Data inladen ===

    train_df, test_df = Split_data.split_data(train=(1,2,3,4), test=(5,))

    RandomF(X_train=train_df['content'],Y_train=train_df['label'],X_test=test_df['content'],Y_test=test_df['label'],grid=False, m_features_vec=5000)
    # GradBoost(X_train=train_df['content'],Y_train=train_df['label'],X_test=test_df['content'],Y_test=test_df['label'],grid=False)

    # # === 2. Vectorisatie ===
    # X_train, X_test, y_train, y_test, vectorizer = vectorize_train_test(
    #     train_df,
    #     test_df,
    #     ngram_range=(1, 1),    # (1, 2) voor unigrams + bigrams
    #     use_tfidf=False,       # zet True voor TF-IDF
    #     max_features=5000,
    # )

    # # === 3. Model trainen ===
    # model, best_alpha, cv_score = train_single_tree(X_train, y_train)
    # print("Beste ccp_alpha:", best_alpha)
    # print(f"Cross-validation score (F1): {cv_score:.4f}")

    # # === 4. Evalueren ===
    # scores = evaluate_model(model, X_test, y_test)
    # print("\n=== Testresultaten (fold 5) ===")
    # for k, v in scores.items():
    #     print(f"{k.capitalize():>9}: {v:.4f}")

    # # === 5. Belangrijkste woorden ===
    # top_words = get_top_features(model, vectorizer, top_n=10)
    # print("\n=== Belangrijkste woorden ===")
    # for word, importance in top_words:
    #     print(f"{word:20s} {importance:.6f}")

if __name__ == "__main__":
    main()
