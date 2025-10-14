#berekent accuracy, precision, recall, f1-score
<<<<<<< Updated upstream
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
=======
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
>>>>>>> Stashed changes

def evaluate_model(model, X_test, y_test):
    """Bereken accuracy, precision, recall en F1-score op de testset."""
    y_pred = model.predict(X_test)
<<<<<<< Updated upstream
=======

    print("accuracy: ", accuracy_score(y_test, y_pred))
    # print("precision: ", precision_score(y_test, y_pred))
    # print("recall: ", recall_score(y_test, y_pred))
    # print("f1: ", f1_score(y_test, y_pred))
    print("Table: ", classification_report(y_test, y_pred))
    print("confusion matrix: ", confusion_matrix(y_test, y_pred))
    
>>>>>>> Stashed changes
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
<<<<<<< Updated upstream
    }
=======
        "confusion matrix": confusion_matrix(y_test, y_pred)
    }


>>>>>>> Stashed changes
