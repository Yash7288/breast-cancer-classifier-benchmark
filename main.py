from dataset_loader import load_data
from preprocessor import preprocessing
from models import models
from evaluate import evaluate_models, print_results


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocessing(X, y)
    results = evaluate_models(models,X_train,X_test,y_train,y_test)
    print_results(results)
if __name__ == "__main__":
    main()