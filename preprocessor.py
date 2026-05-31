import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocessing(X, y):


    y = np.where(y == 'M', 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    imr = SimpleImputer(missing_values=np.nan,strategy='mean')
    X_train = imr.fit_transform(X_train)
    X_test = imr.transform(X_test)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
    