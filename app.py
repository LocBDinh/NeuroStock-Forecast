from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from keras.models import load_model
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
        # NOTE: Need to tokenize the ticker input and call the model
        ticker = request.form.get['ticker']
        stock = yf.Ticker(ticker)
    except:
        return 'Model could not be found.'
    return render_template('index.html', ticker=ticker)

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
