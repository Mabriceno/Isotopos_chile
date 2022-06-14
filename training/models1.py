from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model


class ModelTree():
    'Arbol de decision'

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None


    def fit(self, depth):
        clf = tree.DecisionTreeRegressor(max_depth = depth)
        clf = clf.fit(self.X_train, self.y_train)
        self.model = clf

    def predict(self, X_test):
        return self.model.predict(X_test)
    

class ModelForest():
    'Random Forest'

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None


    def fit(self, depth):
        clf = RandomForestRegressor( max_depth=depth, random_state=0)
        clf = clf.fit(self.X_train, self.y_train)
        self.model = clf

    def predict(self, X_test):
        return self.model.predict(X_test)

class ModelMLP():
    'MLP'

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None


    def fit(self, activation):
        clf = MLPRegressor(activation = activation, random_state=0, max_iter=10000)
        clf = clf.fit(self.X_train, self.y_train)
        self.model = clf

    def predict(self, X_test):
        return self.model.predict(X_test)
    
class ModelBestLR():
    ''

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
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


    def predict(self, X_test):
        [i,j,k] = self.list_model1
        X_test = np.transpose(X_test)
        X_test = [X_test[i],X_test[j],X_test[k]]
        X_test = np.transpose(X_test)
        return self.model.predict(X_test)


class ModelBestRR():
    ''

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
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
                        self.model = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
                        self.list_model1 = [i,j,k]


    def predict(self, X_test):
        [i,j,k] = self.list_model1
        X_test = np.transpose(X_test)
        X_test = [X_test[i],X_test[j],X_test[k]]
        X_test = np.transpose(X_test)
        return self.model.predict(X_test)


