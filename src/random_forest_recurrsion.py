from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
import pickle
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

    def score_model(self, X, y):
        with open('data/rfr_model.pickle', 'rb') as handle:
            rfr = pickle.load(handle)
        return rfr.score(X, y)

if __name__== '__main__':
    with open('data/Stats.pickle', 'rb') as handle:
        data = pickle.load(handle)

    rfr = RFR()
    data = data.astype(np.float64)
    X = data[::-1,:11]
    y = data[::-1,11]
    print(X[100])
    print(y[100])
    #rfr.find_optimal_hyp(X, y)
    #rfr.train_model(X, y)
    test = rfr.predict_model(X[100:101,:11])
    print(test)
    #print(rfr.score_model(X, y))
