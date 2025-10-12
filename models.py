"""
models.py
----------
Bevat het model en trainingsfunctie voor de Single Classification Tree
uit de op_spam_v1.4 opdracht.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_single_tree(X_train, y_train, cv=5):
    """
    Train een enkele DecisionTreeClassifier met tuning van ccp_alpha via cross-validatie.

    Parameters
    ----------
    X_train : sparse matrix
        Featurematrix van de trainingsteksten.
    y_train : array
        Labels (0 = truthful, 1 = deceptive).
    cv : int, default=5
        Aantal cross-validatie folds.

    Returns
    -------
    model : DecisionTreeClassifier
        Het beste model na tuning.
    best_alpha : float
        De gekozen waarde van ccp_alpha.
    best_score : float
        De beste F1-score tijdens cross-validatie.
    """
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {"ccp_alpha": [0.0, 0.0005, 0.001, 0.005, 0.01]}

    grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_alpha = grid.best_params_["ccp_alpha"]
    best_score = grid.best_score_

    return best_model, best_alpha, best_score


def get_top_features(model, vectorizer, top_n=10):
    """
    Geef de top-n belangrijkste woorden op basis van feature_importances_.

    Parameters
    ----------
    model : DecisionTreeClassifier
        Getraind model.
    vectorizer : CountVectorizer of TfidfVectorizer
        De vectorizer die is gebruikt voor feature-namen.
    top_n : int
        Aantal topwoorden dat je wilt terugzien.

    Returns
    -------
    lijst van tuples (woord, importance)
    """
    import numpy as np

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    feature_names = vectorizer.get_feature_names_out()
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in idx if importances[i] > 0.0]
