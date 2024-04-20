from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# NOTE: Formatted stock info
"""
for key, value in stock.info.items():
     print(key, ':', value)
"""

def train_model(ticker):
    # User input for stock ticker from Flask App
    stock = yf.Ticker(ticker)

    # Set the start and end date
    start_year = "2000"
    end = str(dt.date.today())

    # Download historical data for required stocks
    data = yf.download(ticker, start=(start_year + "-01-01"), end = end, 
                    auto_adjust = True) # Auto_adjust automatically adjusts the stock prices for splits and dividends
    
    # Show User the stock info
    print("\nStock Information:")
    print("------------------")
    print(f"Name: {stock.info['shortName']}")
    print(f"Ticker: {ticker}")
    print(f"Open: {stock.info['open']}")
    print(f"Previous Close: {stock.info['previousClose']}")
    print(f"Volume: {stock.info['volume']}")

    # Not all stocks have all information
    try:
        print(f"Dividend Yield: {stock.info['dividendYield']}")
    except:
        print("Dividend Yield: N/A")

    try:
        print(f"Market Cap: {stock.info['marketCap']}")
    except:
        print("Market Cap: N/A")

    try:
        print(f"52 Week High: {stock.info['fiftyTwoWeekHigh']}")
        print(f"52 Week Low: {stock.info['fiftyTwoWeekLow']}")
    except:
        print("52 Week High: N/A")
        print("52 Week Low: N/A")

    try:
        print(f"PE Ratio: {stock.info['trailingPE']}")
    except:
        print("PE Ratio: N/A")

    try:
        print(f"Earnings Per Share: {stock.info['trailingEps']}")
    except:
        print("Earnings Per Share: N/A")

    # Store the most recent 5 day data
    recent_df = yf.download(ticker, period = "5d") 

    # Store Dividend Yield [Display] and Stock Split History [Display]
    dividends = stock.dividends
    splits = stock.splits

    # Displays 5 year chart of stock
    data = stock.history(period = '5y' )
    data['Close'].plot()
    plt.title(f"{ticker} Stock Prices")
    #plt.show()
    plt.savefig("static/icons/fiveYearPlot.svg", format="svg")
    plt.close()

    # Inserts the data into a pandas dataframe
    data = yf.download(ticker)[['Adj Close']]
    data.reset_index(inplace = True)
    data.drop('Date', axis = 1, inplace = True)

    # Split the data into training and testing data
    split_percent = 0.90
    split_point = round(len (data) * split_percent)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    # Using Scikit-Learn to preprocess data before feeding it into the model
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    # Description: This file contains the functions for preprocessing the time series data
    def timeseries_preprocessing(scaled_train, scaled_test, lags):
        X,Y = [],[]
        for t in range(len(scaled_train) - lags - 1):
            X.append(scaled_train[t:(t + lags), 0])
            Y.append(scaled_train[(t + lags), 0])

        Z,W = [],[]
        for t in range(len(scaled_test) - lags - 1):
            Z.append(scaled_test[t:(t + lags), 0])
            W.append(scaled_test[(t + lags), 0])

        X_train, Y_train, X_test, Y_test = np.array(X), np.array(Y), np.array(Z), np.array(W)

        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

        return X_train, Y_train, X_test, Y_test

    # Preprocess the data
    X_train, Y_train, X_test, Y_test = timeseries_preprocessing(scaled_train, scaled_test, 10)

    # Create the model using Keras
    model = Sequential()
    model.add(LSTM(256,input_shape = (X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mse')

    # Train the model on the training data with 15 epochs [Should be changed to 500+ for better results]
    history = model.fit(x = X_train,y = Y_train, epochs = 15, validation_data = (X_test,Y_test), shuffle = False)

    # Plot the loss and validation loss
    axes = plt.axes()
    axes.plot(pd.DataFrame(model.history.history)['loss'], label = 'Loss')
    axes.plot(pd.DataFrame(model.history.history)['val_loss'], label = 'Validation Loss')
    axes.legend(loc=0)
    axes.set_title('Model Fitting Performance')
    #plt.show()
    plt.savefig("static/icons/lossPlot.svg", format="svg")
    plt.close()

    # Predict the stock prices using the model
    Y_predicted=scaler.inverse_transform(model.predict(X_test))
    Y_true=scaler.inverse_transform(Y_test.reshape(Y_test.shape[0],1))

    # Plot the predicted stock prices vs the true stock prices from data
    axes=plt.axes()
    axes.plot(Y_true, label='True Y')
    axes.plot(Y_predicted, label='Predicted Y')
    axes.legend(loc=0)
    axes.set_title('Prediction Adjustment')
    #plt.show()
    plt.savefig("static/icons/predictionPlot.svg", format="svg")
    plt.close()

    # Calculate the metrics for the model
    Y_predicted = scaler.inverse_transform(model.predict(X_test))
    Y_true = scaler.inverse_transform(Y_test.reshape(Y_test.shape[0], 1))
    print()
    print("Mean Absolute Error: {:.2f}".format(metrics.mean_absolute_error(Y_true, Y_predicted)))
    print("Mean Squared Error: {:.2f}".format(metrics.mean_squared_error(Y_true, Y_predicted)))
    print("Root Mean Squared Error: {:.2f}".format(np.sqrt(metrics.mean_squared_error(Y_true, Y_predicted))))
    print("R2 Score: {:.2f}".format(metrics.r2_score(Y_true, Y_predicted)))
    print()

    # Calculate the model accuracy, precision, recall, and F1 score
    print("Accuracy: {:.2f}".format((1 - (metrics.mean_absolute_error(Y_true, Y_predicted) / Y_true.mean())) * 100))
    print("Precision: {:.2f}".format((1 - (metrics.mean_squared_error(Y_true, Y_predicted) / Y_true.mean())) * 100))
    print("Recall: {:.2f}".format((1 - (np.sqrt(metrics.mean_squared_error(Y_true, Y_predicted)) / Y_true.mean()) * 100)))
    print("F1 Score: {:.2f}".format((1 - (metrics.r2_score(Y_true, Y_predicted) / Y_true.mean()) * 100)))
    print()

    # Save the model
    model.save("neurostock_forecast.keras")
    # Load the model
    try:
        model = tf.keras.models.load_model("neurostock_forecast.keras")
        print("Model loaded successfully.")
    except:
        print("Model could not be loaded.")
        