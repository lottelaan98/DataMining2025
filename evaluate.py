"""
Evaluate a fitted classifier on a test set by computing accuracy, the full classification report, and the confusion matrix.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
   """
    Run predictions and print core metrics + summary tables.

    Returns dict
        Dictionary with accuracy, classification report (string), and confusion matrix (ndarray).
    """
    # Predict labels for the test set
    y_pred = model.predict(X_test)

    print("accuracy: ", accuracy_score(y_test, y_pred))

    # Detailed per-class report and confusion matrix
    print("Table: ", classification_report(y_test, y_pred))
    print("confusion matrix: ", confusion_matrix(y_test, y_pred))
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "Table: ": classification_report(y_test, y_pred),
        "confusion matrix": confusion_matrix(y_test, y_pred)
    }

