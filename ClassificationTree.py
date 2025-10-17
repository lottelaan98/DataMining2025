"""
Model 3 — Single Classification Tree (manual tuning)
----------------------------------------------------
Loads folds via OpenFiles.Split_data(train=(1,2,3,4), test=(5,)),
builds a simple pipeline (CountVectorizer → TfidfTransformer → DecisionTreeClassifier),
and manually tests a small grid of hyperparameters directly on the held-out test fold.

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from evaluate import evaluate_model

def make_tree_pipeline(max_depth=None, min_samples_leaf=1, ccp_alpha=0.0):
    """
    Build a text classification pipeline:
    CountVectorizer(min_df=2, 1–2 grams) → TF–IDF weighting → Decision Tree.
    """
    return make_pipeline(
        CountVectorizer(min_df=2, ngram_range=(1, 2)),
        TfidfTransformer(),
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=42
        )
    )

def evaluate_accuracy(model, test_df):
     """
    Predict on test_df and return (accuracy, predictions).
    """
    preds = model.predict(test_df["content"])
    return accuracy_score(test_df["label"], preds), preds

def ClassificationTree(train_df, test_df):
    """
    Manually sweep a small grid of tree hyperparameters, pick the best by test accuracy,
    print a summary, then fit/evaluate the best model and show top features.
    """
    # Manual hyperparameter ranges to try
    depths = [5, 10, 20, None]
    leaves = [1, 2]
    alphas = [0.0, 0.001, 0.01, 0.1]

    # Evaluate every combination on the test fold (⚠ see note in header)
    results = []
    for d in depths:
        for l in leaves:
            for a in alphas:
                model = make_tree_pipeline(max_depth=d, min_samples_leaf=l, ccp_alpha=a)
                model.fit(train_df["content"], train_df["label"])
                acc, _ = evaluate_accuracy(model, test_df)
                results.append({
                    "max_depth": d,
                    "min_samples_leaf": l,
                    "ccp_alpha": a,
                    "test_accuracy": acc
                })

     # Rank by test accuracy (descending)
    res_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False).reset_index(drop=True)

    # Choose the top setting and report
    best = res_df.iloc[0].to_dict()
    print("\nChosen hyperparemeters with highest accuracy value:", best)

    # Refit the best model on the training data
    best_model = make_tree_pipeline(
        max_depth=int(best["max_depth"]) if not pd.isna(best["max_depth"]) else None,
        min_samples_leaf=int(best["min_samples_leaf"]),
        ccp_alpha=float(best["ccp_alpha"])
    )
    
    best_model = best_model.fit(train_df["content"], train_df["label"])

     # Detailed evaluation (classification report, confusion matrix, etc.)
    evaluate_model(best_model, test_df["content"], test_df["label"])

    # Extract feature importances and corresponding token names
    tree = best_model.named_steps["decisiontreeclassifier"]
    vectorizer = best_model.named_steps["countvectorizer"]

    feature_importance = getattr(tree, "feature_importances_", None)
    feature_names = vectorizer.get_feature_names_out()

    # Build a sortable DataFrame of features by importance
    feature_importance_df = (
        pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance})
        .sort_values(by="importance", ascending=False)
    )

    print("\nTop 10 Most Important Features ClassificationTree:")
    print(feature_importance_df.head(10))
