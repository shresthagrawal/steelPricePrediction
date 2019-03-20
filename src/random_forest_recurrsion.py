from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
import pickle

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
        prediction = rfr.predict(X)
        return prediction

    def score_model(self, X, y):
        with open('data/rfr_model.pickle', 'rb') as handle:
            rfr = pickle.load(handle)
        return rfr.score(X, y)

iris = load_iris()
rfr = RFR()
#rfr.find_optimal_hyp(iris['data'], iris['target'])
#rfr.train_model(iris['data'], iris['target'])
print(rfr.score_model(iris['data'], iris['target']))
