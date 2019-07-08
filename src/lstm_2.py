import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM as tf_LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import load_model

import pickle


class LSTM:
    def __init__(self):
        with open('data/Stats2019_with_cr.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
        self.data = self.data[::-1]
        self.data = self.data.astype(numpy.float64)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(self.data)
        self.look_back = 15
        self.features = 13
        self.y_pos =11
        self.future = 5

    def create_dataset(self, dataset, look_back, dim, future):
    	dataX, dataY = [], []
    	for i in range(len(dataset)-look_back-future):
    		dataX.append(dataset[i:(i+look_back)])
    		dataY.append(dataset[i + look_back + future, self.y_pos])
    	return numpy.array(dataX), numpy.array(dataY)

    def train_model(self, X, Y):
        # reshape input to be [samples, time steps, features]
        #X, Y = self.create_dataset(self.data, self.look_back, self.features)
        # create and fit the LSTM network
        model = Sequential()
        model.add(tf_LSTM(8, input_shape=(self.look_back, self.features)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=100, batch_size=1, verbose=2)
        model.save('lstm.h5')  # creates a HDF5 file 'my_model.h5'
        return True

    def predict_model(self, X):
        model = load_model('lstm.h5')
        Y_predicted = model.predict(X)
        # invert predictions
        Y_predicted = self.scaler.inverse_transform(Y_predicted)
        return Y

    def score_model(self, X, Y):
        #X, Y = self.create_dataset(self.data, self.look_back, self.features)
        model = load_model('lstm.h5')
        Y_predicted = model.predict(X)

        length = len(Y)
        test = numpy.empty([length, self.features])
        test[:,self.y_pos] = numpy.reshape(Y_predicted, (length))
        Y_predicted = (self.scaler.inverse_transform(test))[:,self.y_pos]

        test[:,self.y_pos] = numpy.reshape(Y, (length))
        Y = (self.scaler.inverse_transform(test))[:,self.y_pos]

        # calculate root mean squared error
        print(Y[0], Y_predicted[0])
        trainScore = math.sqrt(mean_squared_error(Y, Y_predicted))
        print('Train Score: %.2f RMSE' % (trainScore))

        plt.plot(Y)
        plt.plot(Y_predicted)
        plt.show()

    def predict_future(self, uptil_when):
        X, _ = self.create_dataset(self.data, self.look_back, self.features)
        X = X[len(X)-1:]
        #print(X[:,1:])

        model = load_model('lstm.h5')
        Y_predicted = []
        for future in range(uptil_when):
            prediction = model.predict(X)
            Y_predicted.append(prediction[0])
            X = X[:,1:]
            prediction = numpy.reshape(prediction,(1,1,self.features))
            X = numpy.append(X, prediction, 1)

        Y_predicted = (self.scaler.inverse_transform(Y_predicted))
        return Y_predicted





if __name__== '__main__':
    lstm = LSTM()
    X, Y = lstm.create_dataset(lstm.data, lstm.look_back, lstm.features, lstm.future)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False)
    #x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0, shuffle=True)
    #print(x_train)
    #lstm.train_model(x_train, y_train)
    #lstm.score_model(X, Y)
    max = 0
    data = lstm.scaler.inverse_transform(lstm.data)
    for row in data:
        if row[lstm.y_pos]>max:
            max = row[lstm.y_pos]
            print(row[0],row[1])
    """

    model = load_model('lstm.h5')
    Y_predicted = model.predict(X)
    length = len(Y)
    test = numpy.empty([length, lstm.features+1])
    test[:,lstm.features] = numpy.reshape(Y_predicted, (length))
    Y_predicted = (lstm.scaler.inverse_transform(test))[:,lstm.features]

    for val in Y_predicted:
        print(str(val)+",")
    """
    #lstm.score_model()
    #future = lstm.predict_future(10)
    #numpy.savetxt("results/2019LSTM2.csv", future, delimiter=",")
