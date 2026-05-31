from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

models = {
    "Perceptron": Perceptron(eta0 = 0.01, max_iter = 1000),
    "Logistic Regression": LogisticRegression(),
    "Linear SVM": SVC(kernel="linear"),
    "RBF SVM": SVC(kernel="rbf"),
    "Decision Tree": DecisionTreeClassifier(random_state = 1),
    "Random Forest": RandomForestClassifier(random_state = 1),
    "KNN" : KNeighborsClassifier(n_neighbors = 5)
}