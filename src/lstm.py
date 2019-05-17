import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM as tf_LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model

import pickle


class LSTM:
    def __init__(self):
        with open('data/Stats2019.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
        self.data = self.data[::-1]
        self.data = self.data.astype(numpy.float64)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(self.data)
        self.look_back = 15
        self.features = 11

    def create_dataset(self, dataset, look_back, dim):
    	dataX, dataY = [], []
    	for i in range(len(dataset)-look_back-1):
    		a = dataset[i:(i+look_back), :dim]
    		dataX.append(a)
    		dataY.append(dataset[i + look_back, dim])
    	return numpy.array(dataX), numpy.array(dataY)

    def train_model(self):
        # reshape input to be [samples, time steps, features]
        X, Y = self.create_dataset(self.data, self.look_back, self.features)
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

    def score_model(self):
        X, Y = self.create_dataset(self.data, self.look_back, self.features)
        model = load_model('lstm.h5')
        Y_predicted = model.predict(X)
        # invert predictions
        length = len(Y)
        test = numpy.empty([length, self.features+1])
        test[:,self.features] = numpy.reshape(Y_predicted, (length))
        Y_predicted = (self.scaler.inverse_transform(test))[:,self.features]

        test[:,self.features] = numpy.reshape(Y, (length))
        Y = (self.scaler.inverse_transform(test))[:,self.features]

        #Y = scaler.inverse_transform(Y)
        # calculate root mean squared error
        print(Y[0], Y_predicted[0])
        trainScore = math.sqrt(mean_squared_error(Y, Y_predicted))
        print('Train Score: %.2f RMSE' % (trainScore))

        plt.plot(Y)
        plt.plot(Y_predicted)
        plt.show()


if __name__== '__main__':
    lstm = LSTM()
    #lstm.train_model()
    X, Y = lstm.create_dataset(lstm.data, lstm.look_back, lstm.features)
    model = load_model('lstm.h5')
    Y_predicted = model.predict(X)
    length = len(Y)
    test = numpy.empty([length, lstm.features+1])
    test[:,lstm.features] = numpy.reshape(Y_predicted, (length))
    Y_predicted = (lstm.scaler.inverse_transform(test))[:,lstm.features]

    for val in Y_predicted:
        print(str(val)+",")
    lstm.score_model()
