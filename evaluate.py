#berekent accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """Bereken accuracy, precision, recall en F1-score op de testset."""
    y_pred = model.predict(X_test)

    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("Table: ", classification_report(y_test, y_pred))
    print("confusion matrix: ", confusion_matrix(y_test, y_pred))
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "Table: ": classification_report(y_test, y_pred),
        "confusion matrix": confusion_matrix(y_test, y_pred)
    }

