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
    X_train: training data, dus tekst van reviews
    Y_train: training labels
    m_depth: max depth van de trees 
    lambda: learning rate, parameter voor gradient boosting, 
    n_trees: hoeveel trees bagging achter elkaar doet, overfit niet snel
    grid: Of we wel of niet een grid search doen, als grid True is doen we dat wel en returnen we het beste model dat daar uitkomt
          de default values die al ingevuld zijn (bij if not grid) zijn het resultaat van een gridsearch en behalen de hoogste accuracy
    """

    vectorizer = TfidfVectorizer(max_features=m_features_vec, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_train)
    print(f"Shape of INPUT: {X.shape}")

    if grid:
        bosgrid = GBR(max_features='sqrt',random_state=0,max_leaf_nodes=60)

        params = { 'max_depth': [15,20,25,None],
                'learning_rate': [0.01,0.001],
                'n_estimators': [250]
                    }
        
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
        bos = GBR(n_estimators=n_trees,max_features="sqrt",max_depth=m_depth,random_state=0,learning_rate=lam)
        bos.fit(X,Y_train)

        evaluate_model(bos, vectorizer.transform(X_test), Y_test)

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
