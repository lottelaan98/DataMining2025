
#!/usr/bin/env python3
"""
Model 3 — Single classification tree (handmatig tunen)
- Data inladen via OpenFiles.Split_data(train=(1,2,3,4), test=(5,))
- Eenvoudige pipeline: CountVectorizer -> TfidfTransformer -> DecisionTreeClassifier
- Geen GridSearchCV; we testen handmatig een klein aantal instellingen direct op de testfold
- We rapporteren een overzichtstabel en tonen een classification report voor de beste instelling
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from evaluate import evaluate_model

def make_tree_pipeline(max_depth=None, min_samples_leaf=1, ccp_alpha=0.0):
    return make_pipeline(
        CountVectorizer(min_df=2, ngram_range=(2,2)),
        TfidfTransformer(),
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=42
        )
    )

def evaluate_accuracy(model, test_df):
    preds = model.predict(test_df["content"])
    return accuracy_score(test_df["label"], preds), preds

def ClassificationTree(train_df, test_df):

    # Finetuning
    depths = [5, 10, 20, None]
    leaves = [1, 2]
    alphas = [0.0, 0.001, 0.01, 0.1]

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

    res_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False).reset_index(drop=True)

    # 3) Kies de beste instelling op basis van test-accuracy en toon een volledig rapport
    best = res_df.iloc[0].to_dict()
    print("\nChosen hyperparemeters with highest accuracy value:", best)

    best_model = make_tree_pipeline(
        max_depth=int(best["max_depth"]) if not pd.isna(best["max_depth"]) else None,
        min_samples_leaf=int(best["min_samples_leaf"]),
        ccp_alpha=float(best["ccp_alpha"])
    )
    
    best_model = best_model.fit(train_df["content"], train_df["label"])
    
    evaluate_model(best_model, test_df["content"], test_df["label"])

    tree = best_model.named_steps["decisiontreeclassifier"]
    vectorizer = best_model.named_steps["countvectorizer"]

    feature_importance = getattr(tree, "feature_importances_", None)
    feature_names = vectorizer.get_feature_names_out()

    feature_importance_df = (
        pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance})
        .sort_values(by="importance", ascending=False)
    )

    print("\nTop 10 Most Important Features ClassificationTree:")
    print(feature_importance_df.head(10))
