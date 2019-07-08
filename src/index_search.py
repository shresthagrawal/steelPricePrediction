import csv

import matplotlib.pyplot as plt

import numpy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize


from utils import numpy_to_csv


def create_timestamped_map(file_name):
    raw_data = {}
    with open(file_name) as csvfile:
        dict = csv.DictReader(csvfile)
        for row in dict:
            raw_data[row['year'] + row['week']] = row
    return raw_data


def join_odered_list(row1, row2, columns_to_ignore):
    for key, val in row2.items():
        if key not in columns_to_ignore:
            row1[key] = val
    return row1


def get_data(file_names):
    maps = []
    columns_to_ignore = ['week', 'year', 'monday', 'sunday']
    for x in range(0, len(file_names) - 1):
        maps.append(create_timestamped_map(file_names[x]))
    raw_data = []
    with open(file_names[len(file_names) - 1]) as csvfile:
        dict = csv.DictReader(csvfile)
        for row in dict:
            for map in maps:
                if (row['year'] + row['week']) in map:
                    row = join_odered_list(
                        row,
                        map[row['year'] + row['week']],
                        columns_to_ignore)
            raw_data.append(row)
    return raw_data


def get_columns(raw_data, x_column, y_column, add_timestamp, previous_cnt=0):
    x = []
    y = []
    for index in range(previous_cnt, len(raw_data)):
        check = True
        for time in range(0, previous_cnt + 1):
            if(
                (x_column not in raw_data[index - time]) or
                (y_column not in raw_data[index - time]) or
                raw_data[index - time][x_column] == '-' or
                    raw_data[index - time][y_column] == '-'):
                check = False
                break
        if check is False:
            continue
        row_x = []
        if add_timestamp:
            row_x.append(float(raw_data[index]['week']))
            row_x.append(float(raw_data[index]['year']))
        for time in range(0, previous_cnt + 1):
            row_x.append(float(raw_data[index - time][x_column]))
        x.append(row_x)
        y.append(float(raw_data[index][y_column]))
    x = numpy.array(x, dtype=float)
    y = numpy.array(y, dtype=float)
    return x, y

def find_optimal_param_rfr(x, y):
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3, 7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=True, n_jobs=-1)

    grid_result = gsc.fit(x, y)
    optimal_params = grid_result.best_params_
    return optimal_params

def train_and_score(x, y, degree):
    # model = make_pipeline(PolynomialFeatures(degree), Ridge(tol=1, copy_X=True,
    #                      normalize=True))
    model = make_pipeline(LinearRegression(normalize=True))
    # optimal_params = find_optimal_param_rfr(x, y)
    # model = RandomForestRegressor(
    #     max_depth=optimal_params["max_depth"],
    #     n_estimators=optimal_params["n_estimators"],
    #     random_state=False,
    #     verbose=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_predicted = model.predict(X_test)
    return model, y_predicted, score


def create_corelation_matrix(file_names, column_names):
    data = get_data(file_names)
    result = []
    for column_name_x in column_names:
        result_row = []
        for column_name_y in column_names:
            # Finds the corelation between y for x
            X, y = get_columns(data, column_name_x, column_name_y, True, 2)
            if(len(X) < 50):
                score = 0.00
            else:
                model, y_predicted, score = train_and_score(X, y, 0)
            result_row.append(score)
        result_row.append(numpy.sum(result_row))
        result.append(result_row)
    header = column_names[:]
    header.append('sum')
    index = column_names[:]
    return result, header, index


def plot_matrix(matrix, header, index):
    sub = plt.subplot()
    # sub.title('Corelation Matrix')
    # Set Precision to 3
    matrix = numpy.around(matrix, decimals=3)
    sub.xaxis.set_visible(False)
    sub.yaxis.set_visible(False)
    table = sub.table(
        cellText=matrix,
        rowLabels=header,
        colLabels=index,
        loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(4)
    table.scale(1, 1)

file_names = [
    'data/pellet_historical_data/pellet_cnf.csv',
    'data/pellet_historical_data/pellet_fob.csv',
    'data/pellet_historical_data/pellet_domestic.csv']
column_names = ['Barbil', 'Bellary', 'Jajpur', 'Durgapur',
                'Jamshedpur', 'Jharsugda', 'Kandla', 'Keonjhar', 'Raipur']
other_columns = [
    "India,Prices BF GRADE, 6-20MM, Fe 64%",
    "China,Prices BF GRADE, 6-20MM, Fe 63.5%",
    "China,Prices BF GRADE, 6-20MM, Fe 65%",
    "China,Premium BF GRADE, 5-20MM, 65%",
    "India,Prices BF GRADE, 6-20MM, Fe 64%"]

matrix, header, index = create_corelation_matrix(
    file_names,
    column_names + other_columns)
numpy_to_csv(matrix, header, index,
             'data/results/linear_regression_history2.csv')
# plot_matrix(matrix, column_names+other_columns)
# plt.show()
