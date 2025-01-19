from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return '''
        <h1>House Price Prediction</h1>
        <form action="/predict" method="POST">
            Area (sq ft): <input type="text" name="area"><br>
            Rooms: <input type="text" name="rooms"><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs
    area = float(request.form['area'])
    rooms = int(request.form['rooms'])
    
    # Make the prediction
    prediction = model.predict([[area, rooms]])

    # Return the result
    return jsonify({
        'predicted_price': prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True)
