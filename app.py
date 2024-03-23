from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

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
    except:
        return 'Model could not be found.'
    return render_template('predictions.html', ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True, host='', port=1337) # Debug mode is on and allows for reloading the server when changes are made
