# MADE TWO FILES SO THE MODELS ARE SEPERATED 
"""
# models.py
# ----------
# Bevat het model en trainingsfunctie voor de Single Classification Tree
# uit de op_spam_v1.4 opdracht.
# """

# from sklearn.tree import DecisionTreeClassifier
# import numpy as np
# import pandas as pd
# from sklearn.tree import (DecisionTreeClassifier as DTC ,
#                           DecisionTreeRegressor as DTR ,
#                            plot_tree ,
#                            export_text)
# from sklearn.metrics import (accuracy_score ,
#                              log_loss,
#                              classification_report)
# from sklearn.ensemble import (RandomForestClassifier as RF, #For the random forest
#                               GradientBoostingClassifier as GBR)
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import GridSearchCV


# def train_single_tree(X_train, y_train, cv=5):
#     """
#     Train een enkele DecisionTreeClassifier met tuning van ccp_alpha via cross-validatie.

#     Parameters
#     ----------
#     X_train : sparse matrix
#         Featurematrix van de trainingsteksten.
#     y_train : array
#         Labels (0 = truthful, 1 = deceptive).
#     cv : int, default=5
#         Aantal cross-validatie folds.

#     Returns
#     -------
#     model : DecisionTreeClassifier
#         Het beste model na tuning.
#     best_alpha : float
#         De gekozen waarde van ccp_alpha.
#     best_score : float
#         De beste F1-score tijdens cross-validatie.
#     """
#     model = DecisionTreeClassifier(random_state=42)
#     param_grid = {"ccp_alpha": [0.0, 0.0005, 0.001, 0.005, 0.01]}

#     grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1")
#     grid.fit(X_train, y_train)

#     best_model = grid.best_estimator_
#     best_alpha = grid.best_params_["ccp_alpha"]
#     best_score = grid.best_score_

#     return best_model, best_alpha, best_score


# def get_top_features(model, vectorizer, top_n=10):
#     """
#     Geef de top-n belangrijkste woorden op basis van feature_importances_.

#     Parameters
#     ----------
#     model : DecisionTreeClassifier
#         Getraind model.
#     vectorizer : CountVectorizer of TfidfVectorizer
#         De vectorizer die is gebruikt voor feature-namen.
#     top_n : int
#         Aantal topwoorden dat je wilt terugzien.

#     Returns
#     -------
#     lijst van tuples (woord, importance)
#     """
#     import numpy as np

#     importances = getattr(model, "feature_importances_", None)
#     if importances is None:
#         return []

#     feature_names = vectorizer.get_feature_names_out()
#     idx = np.argsort(importances)[::-1][:top_n]
#     return [(feature_names[i], float(importances[i])) for i in idx if importances[i] > 0.0]


# def RandomF(X_train,Y_train,X_test,Y_test,m_depth_bos=23,m_features_vec=5000,grid = False):
#     """ 
#     X_train: training data, dus tekst van reviews
#     Y_train: training labels
#     m_features: parameter voor random forest
#     grid: Of we wel of niet een grid search doen, als grid True is doen we dat wel en returnen we het beste model dat daar uitkomt
#           de default values die al ingevuld zijn (bij if not grid) zijn het resultaat van een gridsearch en behalen de hoogste accuracy
#     """

#     #Preprocessing: de vectorizer convert de tekst naar een matrix van token counts met parameters als wel of niet stop words, en een minimale frequency
#     vectorizer = TfidfVectorizer(max_features=m_features_vec, ngram_range=(1,2)) #max_features neemt alleen maar de top x features mee  
#     X = vectorizer.fit_transform(X_train)
#     print(f"Shape of INPUT: {X.shape}")
#     if not grid:
#         bos = RF(n_estimators=250,max_features="sqrt",max_depth=m_depth_bos,random_state=0, min_samples_split=2, oob_score=True,criterion='entropy',max_leaf_nodes=60,verbose=0)
#         bos.fit(X,Y_train)
#         preds = bos.predict(vectorizer.transform(X_test))
#         intpreds = np.round(preds).astype(int)
#         accuracy = accuracy_score(Y_test,intpreds)
#         print(f"ACCURACY: {accuracy}")
#         print(classification_report(Y_test, intpreds))
#         importances = bos.feature_importances_
#         feature_names = vectorizer.get_feature_names_out()

#         feature_importance_df = pd.DataFrame({
#             'feature': feature_names,
#             'importance': importances
#         })

#         feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

#         print("\nTop 20 Most Important Features:")
#         print(feature_importance_df.head(20))
#         return bos


#     else:
#         bosgrid = RF(max_features="sqrt",random_state=0,n_estimators=250, criterion='entropy',max_leaf_nodes=60)

#         params = { 'max_depth': [20,23,30,None],
#                     #'criterion': ['gini', 'entropy', 'log_loss'] 
#                     #'max_leaf_nodes': [30,40,60,None]          
#                     }
        
#         grid_search = GridSearchCV(estimator=bosgrid, param_grid=params,
#                             cv=3, n_jobs=-1, verbose=0)

#         grid_search.fit(X,Y_train)

#         # Get the best parameters
#         print(f"\nBest parameters found with {m_features_vec}: {grid_search.best_params_}")

#         """ Best parameters found with 5000: {'criterion': 'entropy', 'max_depth': 23}
#             0.89375"""
#         # Use the best estimator for predictions
#         best_rf = grid_search.best_estimator_
#         predictions = best_rf.predict(vectorizer.transform(X_test))
#         accuracy = accuracy_score(Y_test,predictions)
#         print(accuracy)
#         return best_rf                              


# def GradBoost(X_train,Y_train,X_test,Y_test,m_features_vec=5000,m_depth=15,lam=0.01,n_trees=500,grid = False):
#     """ 
#     X_train: training data, dus tekst van reviews
#     Y_train: training labels
#     m_depth: max depth van de trees 
#     lambda: learning rate, parameter voor gradient boosting, 
#     n_trees: hoeveel trees bagging achter elkaar doet, overfit niet snel
#     grid: Of we wel of niet een grid search doen, als grid True is doen we dat wel en returnen we het beste model dat daar uitkomt
#           de default values die al ingevuld zijn (bij if not grid) zijn het resultaat van een gridsearch en behalen de hoogste accuracy
#     """

#     vectorizer = TfidfVectorizer(max_features=m_features_vec, ngram_range=(1,2))
#     X = vectorizer.fit_transform(X_train)
#     print(f"Shape of INPUT: {X.shape}")

#     if grid:
#         bosgrid = GBR(max_features='sqrt',random_state=0,max_leaf_nodes=60)

#         params = { 'max_depth': [15,20,25,None],
#                 'learning_rate': [0.01,0.001],
#                 'n_estimators': [250]
#                     }
        
#         grid_search = GridSearchCV(estimator=bosgrid, param_grid=params,
#                             cv=3, n_jobs=-1, verbose=0)

#         grid_search.fit(X,Y_train)

#         # Get the best parameters
#         print(f"\nBest parameters found: {grid_search.best_params_}")

#         # Use the best estimator for predictions
#         best_rf = grid_search.best_estimator_
#         predictions = best_rf.predict(vectorizer.transform(X_test))
#         accuracy = accuracy_score(Y_test,predictions)
#         print(accuracy)
        
#         """Best parameters found: {'learning_rate': 0.01, 'max_depth': 15, 'n_estimators': 500}
#         0.84375 """ 
#         return best_rf

#     else:
#         bos = GBR(n_estimators=n_trees,max_features="sqrt",max_depth=m_depth,random_state=0,learning_rate=lam)
#         bos.fit(X,Y_train)
#         preds = bos.predict(vectorizer.transform(X_test))
#         intpreds = np.round(preds).astype(int)
#         accuracy = accuracy_score(Y_test,intpreds)
#         print(f"ACCURACY: {accuracy}")
#         print(classification_report(Y_test, intpreds))
#         importances = bos.feature_importances_
#         feature_names = vectorizer.get_feature_names_out()

#         feature_importance_df = pd.DataFrame({
#             'feature': feature_names,
#             'importance': importances
#         })

#         feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

#         print("\nTop 20 Most Important Features:")
#         print(feature_importance_df.head(20))
#         return bos
