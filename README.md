# Breast Cancer Classification Benchmark

A machine learning benchmark project comparing multiple classification algorithms on the Wisconsin Breast Cancer Dataset using Scikit-Learn.

## Models Evaluated

- Perceptron
- Logistic Regression
- Linear SVM
- RBF SVM
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

## Preprocessing

- Train-Test Split
- Missing Value Imputation
- Feature Standardization
- Data Leakage Prevention

## Results

| Model | Accuracy | F1 Score |
|---------|---------:|---------:|
| Perceptron | 99.42% | 99.21% |
| Logistic Regression | 97.08% | 96.00% |
| Linear SVM | 96.49% | 95.08% |
| KNN | 96.49% | 95.08% |
| RBF SVM | 95.91% | 94.21% |
| Random Forest | 95.91% | 94.21% |
| Decision Tree | 91.23% | 87.80% |

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn

## Run

bash pip install -r requirements.txt python main.py 
