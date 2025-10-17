"""
Train and evaluate a Gradient Boosting model on text data.

Pipeline:
1) Vectorize reviews with TF–IDF (unigrams + bigrams, capped by max_features).
2) Train GradientBoostingClassifier either:
   - with fixed hyperparameters (default: best values from prior tuning), or
   - with GridSearchCV if `grid=True` to re-tune.
3) Evaluate the model on the test set (accuracy, precision, recall, F1).
4) Print top 10 most important features by information gain.
"""

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model

def GradBoost(X_train,Y_train,X_test,Y_test,m_features_vec=5000,m_depth=15,lam=0.01,n_trees=500,grid = False):
    """
    Train and evaluate a Gradient Boosting classifier.

    Parameters:
      X_train : list-like
          Training text documents.
      Y_train : array-like
          Training labels.
      X_test : list-like
          Test text documents.
      Y_test : array-like
          Test labels.
      m_features_vec : int, optional
          Max number of features in TF–IDF vectorizer (default=5000).
      m_depth : int or None, optional
          Maximum depth of individual trees (default=15).
      lam : float, optional
          Learning rate for boosting (default=0.01).
      n_trees : int, optional
          Number of boosting iterations / trees (default=500).
      grid : bool, optional
          If True, perform GridSearchCV to tune hyperparameters.
          If False, train with provided defaults.
    """

    # Step 1: Vectorize input text (unigrams + bigrams)
    vectorizer = TfidfVectorizer(max_features=m_features_vec, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_train)
    print(f"Shape of INPUT: {X.shape}")

    if grid:
        # Base model with fixed max_leaf_nodes, tuned via grid
        bosgrid = GBR(max_features='sqrt',random_state=0,max_leaf_nodes=60)

        # Hyperparameter search space
        params = { 'max_depth': [15,20,25,None],
                'learning_rate': [0.01,0.001],
                'n_estimators': [250]
                    }
        # Grid search with 3-fold CV, parallelized
        grid_search = GridSearchCV(estimator=bosgrid, param_grid=params,
                            cv=3, n_jobs=-1, verbose=0)

        grid_search.fit(X,Y_train)

        # Get the best parameters
        print(f"\nBest parameters found: {grid_search.best_params_}")

        best_rf = grid_search.best_estimator_

        evaluate_model(best_rf, vectorizer.transform(X_test), Y_test)

        """Best parameters found: {'learning_rate': 0.01, 'max_depth': 15, 'n_estimators': 500}
        0.84375 """ 
        return best_rf

    else:
        # Step 2: Train Gradient Boosting with provided defaults (from prior tuning)
        bos = GBR(n_estimators=n_trees,max_features="sqrt",max_depth=m_depth,random_state=0,learning_rate=lam)
        bos.fit(X,Y_train)

        # Step 3: Evaluate model
        evaluate_model(bos, vectorizer.transform(X_test), Y_test)

        # Step 4: Feature importance inspection
        importances = bos.feature_importances_
        feature_names = vectorizer.get_feature_names_out()

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10))
        return bos
