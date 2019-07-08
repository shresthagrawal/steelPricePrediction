from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
# Random Forest Regression
class RFR:

    def find_optimal_hyp(self, X, y):
        # Perform Grid-Search to find optimal Hyper Parameter
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(3, 7),
                'n_estimators': (10, 50, 100, 1000),
            },
            cv=5, scoring='neg_mean_squared_error', verbose=True, n_jobs=-1)

        grid_result = gsc.fit(X, y)
        optimal_params = grid_result.best_params_
        with open('data/rfr_optimal_params.pickle', 'wb') as handle:
            pickle.dump(optimal_params, handle)

    def train_model(self, X, y):
        with open('data/rfr_optimal_params.pickle', 'rb') as handle:
            optimal_params = pickle.load(handle)
        rfr = RandomForestRegressor(
            max_depth=optimal_params["max_depth"],
            n_estimators=optimal_params["n_estimators"],
            random_state=False,
            verbose=True
            )
        rfr.fit(X, y)
        with open('data/rfr_model.pickle', 'wb') as handle:
            pickle.dump(rfr, handle)

    def predict_model(self, X):
        with open('data/rfr_model.pickle', 'rb') as handle:
            rfr = pickle.load(handle)
        prediction = rfr.predict(X)
        return prediction

    def score_model(self, X, Y):
        with open('data/rfr_model.pickle', 'rb') as handle:
            rfr = pickle.load(handle)

        Y_predicted = rfr.predict(X)
        trainScore = math.sqrt(mean_squared_error(Y, Y_predicted))
        print('Train Score: %.2f RMSE' % (trainScore))

        plt.plot(Y,'r')
        plt.plot(Y_predicted, 'k')
        plt.show()
        return rfr.score(X, Y)

if __name__== '__main__':
    with open('data/Stats2019_with_cr.pickle', 'rb') as handle:
        data = pickle.load(handle)

    rfr = RFR()
    data = data.astype(np.float64)
    data = data[::-1]
    # With the current val what is the next 'future' week value
    future = 5
    X = data[:len(data)-future]
    Y = data[future::,11]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False)

    rfr.find_optimal_hyp(x_train, y_train)
    rfr.train_model(x_train, y_train)
    #predicted_y = rfr.predict_model(x_test)
    #for val in predicted_y:
    #    print(str(val)+",")

    print(rfr.score_model(X, Y))
