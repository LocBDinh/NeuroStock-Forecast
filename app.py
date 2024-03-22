import flask
from flask import request, jsonify
import yfinance as yf
import json
import requests
import os

app = flask.Flask(__name__)

@app.route('/api/v1/health', methods=['GET'])
def status():
    return jsonify({'status': 'UP'})

@app.route('/api/v1/echo', methods=['POST'])
def echo():
    data = request.get_json()
    return jsonify(data)

@app.route('/api/v1/stock', methods=['GET'])
def stock():
    ticker = request.args.get('ticker')
    data = yf.Ticker(ticker)
    return jsonify(data.info)

@app.route('/api/v1/stock', methods=['POST'])
def stock_post():
    data = request.get_json()
    ticker = data['ticker']
    stock = yf.Ticker(ticker)
    return jsonify(stock.info)

@app.route('/api/v1/stock', methods=['PUT'])
def stock_put():
    data = request.get_json()
    ticker = data['ticker']
    stock = yf.Ticker(ticker)
    return jsonify(stock.info)

@app.route('/api/v1/stock', methods=['DELETE'])
def stock_delete():
    data = request.get_json()
    ticker = data['ticker']
    stock = yf.Ticker(ticker)
    return jsonify(stock.info)

if __name__ == '__main__':
    pass