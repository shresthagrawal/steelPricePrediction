import json
from flask import Flask, request, Response
from random_forest_recurrsion import RFR
import numpy as np





app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def api_predict():
    rfr = RFR()
    data = json.loads(request.form['data'])
    print(data)
    prediction = rfr.predict_model(data['data'])
    print(prediction)
    return np.array2string(prediction)


if __name__ == '__main__':
    app.run(host= '0.0.0.0',port= '8080')
