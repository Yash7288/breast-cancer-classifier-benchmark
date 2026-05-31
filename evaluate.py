from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        }
    return results
def print_results(results):
    df = pd.DataFrame(results).T
    print(df.to_string())