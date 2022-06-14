from sklearn import tree
from sklearn import linear_model
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

class Model1():
    'Mejor convinacion en regresion'

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.list_model1 = []
        self.model = None

    def fit(self):
        self.X_train = np.transpose(self.X_train)
        r2 = 0
        for i in range(0, len(self.X_train)):
            for j in range(0, len(self.X_train)):
                for k in range(0, len(self.X_train)):
                    model = linear_model.LinearRegression()
                    X = np.array([self.X_train[i],self.X_train[j],self.X_train[k]])
                    X = np.transpose(X)
                    y = self.y_train
                    model.fit(X, y)
                    r2_aux = model.score(X, y)
                    if (i!=k and i!=j and j!=k) and r2_aux>r2:
                        r2 = r2_aux
                        self.model = model
                        self.list_model1 = [i,j,k]

                    

    def predict(self):
        [i,j,k] = self.list_model1
        X_test = np.transpose(self.X_test)
        X_test = [X_test[i],X_test[j],X_test[k]]
        X_test = np.transpose(X_test)
        return self.model.predict(X_test)


class Model2():
    'Arbol de decision'

    def __init__(self, X_train, X_test, y_train, y_test, depth = 5):
        
        self.depth = depth
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None


    def fit(self):
        clf = tree.DecisionTreeRegressor(max_depth = self.depth)
        clf = clf.fit(self.X_train, self.y_train)
        self.model = clf

    def predict(self, X_test = True):
        if X_test:
            return self.model.predict(self.X_test)
        else:
            return self.model.predict(X_test)
    


class Model3():
    'MLP'

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None


    def fit(self):
        regr = MLPRegressor(random_state=1, max_iter=10000).fit(self.X_train, self.y_train)
        self.model = regr

    def predict(self, X_test = True):
        if X_test:
            return self.model.predict(self.X_test)
        else:
            return self.model.predict(X_test)

class Model4():
    'polinomial'

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None


    def fit(self):
        poly = PolynomialFeatures(degree=2)

        X_train = poly.fit_transform(self.X_train)
        self.X_test = poly.fit_transform(self.X_test)
        model = linear_model.LinearRegression()
        model.fit(X_train, self.y_train)
        self.model = model

    def predict(self, X_test = True):
        if X_test:
            return self.model.predict(self.X_test)
        else:
            return self.model.predict(X_test)
