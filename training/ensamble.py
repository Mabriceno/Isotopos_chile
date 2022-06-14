import numpy as np

class Ensamble():

    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        y_list = []
        for model in self.models:
            y_list.append(model.predict(X))
        
        return np.mean(y_list, axis=0 ), np.std(y_list, axis = 0)
        

class Ensamble_diferente_peso():

    def __init__(self, models1, models2):
        self.models1 = models1
        self.models2 = models2
    
    def predict(self, X, peso = 0.5):
        y_list1 = []
        y_list2 = []

        for model in self.models1:
            X_espacial = np.delete(X, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), 1)
            y_list1.append(model.predict(X_espacial))

        for model in self.models2:
            X_clima = np.delete(X, (20,21,22,23,24,25,26,27,29), 1)
            y_list2.append(model.predict(X_clima))

        y1 = np.array(np.mean(y_list1, axis=0 ))
        y2 = np.array(np.mean(y_list2, axis=0 ))

        y1_ = np.array(np.std(y_list1, axis=0 ))
        y2_ = np.array(np.std(y_list2, axis=0 ))

        ym = (y1*peso + y2*(1-peso))
        std = (y1_*peso + y2_*(1-peso))
        
        return ym, std


class Ensamble_diferente_peso_gradiente():

    def __init__(self, models1, models2):
        self.models1 = models1
        self.models2 = models2
    
    def predict(self, X, peso = 0.5):
        y_list1 = []
        y_list2 = []

        for model in self.models1:
            X_espacial = X
            y_list1.append(model.predict(X_espacial))

        for model in self.models2:
            X_clima = np.delete(X, (20,21,22,23,24,25,26,27,29), 1)
            y_list2.append(model.predict(X_clima))

        y1 = np.split(np.array(np.mean(y_list1, axis=0 )), 44)
        y2 = np.split(np.array(np.mean(y_list2, axis=0 )), 44)

        y1_ = np.split(np.array(np.std(y_list1, axis=0 )), 44)
        y2_ = np.split(np.array(np.std(y_list2, axis=0 )), 44)

        pesos = np.arange(0,peso, peso/44)
        ym=[]
        std=[]
        for i in range(0,44):
            ym.append(y1[i]*(1-pesos[i]) + y2[i]*pesos[i])
            std.append(y1_[i]*(1-pesos[i]) + y2_[i]*pesos[i])
        
        return np.array(ym), np.array(std)

