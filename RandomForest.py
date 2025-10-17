"""
RandomForests model

Contains:
1) Training + simple CV tuning for a single Decision Tree (ccp_alpha),
2) Utility to extract top features from a trained tree,
3) Random Forest training/evaluation for text data (with optional GridSearchCV).

"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.tree import (DecisionTreeClassifier as DTC ,
                          DecisionTreeRegressor as DTR ,
                           plot_tree ,
                           export_text)
from sklearn.metrics import (accuracy_score ,
                             log_loss,
                             classification_report)
from sklearn.ensemble import (RandomForestClassifier as RF, #For the random forest
                              GradientBoostingClassifier as GBR)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model


def train_single_tree(X_train, y_train, cv=5):
    """
    Train a single DecisionTreeClassifier and tune `ccp_alpha` via cross-validation.

    Parameters
    ----------
    X_train : array-like or sparse matrix
        Feature matrix for training (e.g., TF–IDF features).
    y_train : array-like
        Training labels (0 = truthful, 1 = deceptive).
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    model : DecisionTreeClassifier
        Best-scoring model after tuning.
    best_alpha : float
        Selected value of `ccp_alpha`.
    best_score : float
        Best cross-validated F1 score.
    """
    # Base model; only ccp_alpha is tuned here
    model = DecisionTreeClassifier(random_state=42)
    param_grid = {"ccp_alpha": [0.0, 0.0005, 0.001, 0.005, 0.01]}

    # F1-based selection to balance precision/recall
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_alpha = grid.best_params_["ccp_alpha"]
    best_score = grid.best_score_

    return best_model, best_alpha, best_score


def get_top_features(model, vectorizer, top_n=10):
    """
        Return the top-N most important tokens based on `feature_importances_`.

        Parameters
        ----------
        model : DecisionTreeClassifier
            Fitted decision tree.
        vectorizer : CountVectorizer or TfidfVectorizer
            Vectorizer used to obtain feature names.
        top_n : int, default=10
            Number of top tokens to return.

        Returns
        -------
        list of (token, importance)
            Sorted by importance (descending); empty list if no importances.
        """
    import numpy as np

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    feature_names = vectorizer.get_feature_names_out()
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in idx if importances[i] > 0.0]


def RandomF(X_train,Y_train,X_test,Y_test,m_depth_bos=23,m_features_vec=5000,grid = False):
    """
    Train and evaluate a Random Forest classifier on text data (TF–IDF features).

    Parameters
    ----------
    X_train : list-like
        Training text documents (raw strings).
    Y_train : array-like
        Training labels.
    X_test : list-like
        Test text documents (raw strings).
    Y_test : array-like
        Test labels.
    m_depth_bos : int or None, default=23
        Max depth for trees in the Random Forest.
    m_features_vec : int, default=5000
        Max number of TF–IDF vocab features (top-K by frequency).
    grid : bool, default=False
        If True, perform GridSearchCV over a small parameter set; otherwise
        train with predefined hyperparameters (from prior tuning).

    Returns
    -------
    model : RandomForestClassifier
        Fitted Random Forest (best estimator when grid=True; fixed config otherwise).
    """

    # 1) Vectorize text as TF–IDF (unigrams + bigrams), limited to top `m_features_vec` tokens
    vectorizer = TfidfVectorizer(max_features=m_features_vec, ngram_range=(1, 2)) #max_features neemt alleen maar de top x features mee  
    X = vectorizer.fit_transform(X_train)
    print(f"Shape of INPUT: {X.shape}")
    if not grid:
        # 2) Train Random Forest with preset hyperparameters (from previous tuning)
        bos = RF(n_estimators=250,max_features="sqrt",max_depth=m_depth_bos,random_state=0, min_samples_split=2, oob_score=True,criterion='entropy',max_leaf_nodes=60,verbose=0)
        
        bos.fit(X,Y_train)

        # 3) Evaluate on test data
        evaluate_model(bos, vectorizer.transform(X_test), Y_test)

        # 4) Show top features by RF importance
        importances = bos.feature_importances_
        feature_names = vectorizer.get_feature_names_out()

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        print("\nTop 10 Most Important Features RandomForest:")
        print(feature_importance_df.head(10))
        return bos


    else:
        # 2-alt) Grid search over a small space (here just max_depth; others fixed)
        bosgrid = RF(max_features="sqrt",random_state=0,n_estimators=250, criterion='entropy',max_leaf_nodes=60)

        params = { 'max_depth': [20,23,30,None],
                    #'criterion': ['gini', 'entropy', 'log_loss'] 
                    #'max_leaf_nodes': [30,40,60,None]          
                    }
        
        grid_search = GridSearchCV(estimator=bosgrid, param_grid=params,
                            cv=3, n_jobs=-1, verbose=0)

        grid_search.fit(X,Y_train)

        # Get the best parameters
        print(f"\nBest parameters found for RandomForest with {m_features_vec}: {grid_search.best_params_}")

        """ Best parameters found with 5000: {'criterion': 'entropy', 'max_depth': 23}
            0.89375"""
        # Use the best estimator for predictions
        best_rf = grid_search.best_estimator_

        evaluate_model(best_rf, vectorizer.transform(X_test), Y_test)

        return best_rf                              


