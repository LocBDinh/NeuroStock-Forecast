from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from keras.models import load_model
from model import train_model
import yfinance as yf
import tensorflow as tf
import json

app = Flask(__name__)
cors = CORS(app)

# NOTE: Need to import the model here
try:
    model = load_model("neurostock_forecast.keras")
    print("Model loaded successfully.")
except:
    print("Model loaded unsuccessfully.")

@app.route('/', methods=['GET'])
@cross_origin()
def status():
    return render_template('index.html')

@app.route('/stock-predictions', methods=['POST', 'GET'])
@cross_origin()
def stock_predictions():
    try:
        # Retrieve the ticker input from the request
        ticker = request.form.get('ticker')
        if not ticker:
            return 'Ticker not provided.'
        print(f" Your Stock Ticker is: {ticker}")
        # Train model using given ticker
        stock_predictions = train_model(ticker)
        # Return the result to the client
        return render_template('predictions.html', stock_predictions=stock_predictions)
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/data/tickers')
def get_tickers():
    try:
        with open('static/tickers.json', 'r') as file:
            tickers_data = json.load(file)
        return jsonify(tickers_data)
    except FileNotFoundError:
        return "File not found"

if __name__ == '__main__':
    app.run(debug=True, host='', port=1337) # Debug mode is on and allows for reloading the server when changes are made
