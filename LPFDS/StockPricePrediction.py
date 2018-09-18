import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


# plt.switch_backend('TkAgg')

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('/')[2]))
            prices.append(float(row[1]))
    print(dates)
    return
#End get_data

def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear Model')
    plt.plot(dates, svr_poly.predict(dates), color='red', label='Polynomial Model')
    plt.plot(dates, svr_rbf.predict(dates), color='blue', label='RBF Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_poly.predict(x)[0]
#End predict_price

get_data('tesla.csv')
predicted_price = predict_price(dates, prices, 29)
print(predicted_price)