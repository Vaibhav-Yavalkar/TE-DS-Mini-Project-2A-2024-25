from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
import random

app = Flask(__name__)

# Load the trained model from banglore_home_prices_model.pickle
try:
    with open(r'C:\Users\korde\PycharmProjects\HousePricePrediction\banglore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# Load the feature columns from columns.json
with open('columns.json', 'r') as f:
    data_columns = json.load(f)
    feature_columns = data_columns['data_columns']

# Load the historical prices data to train the future price model
historical_data_path = r'C:\Users\korde\PycharmProjects\HousePricePrediction\bengaluru_house_prices.csv'
df = pd.read_csv(historical_data_path)

# Prepare the data for the future price model
X_future = df[['price 2021', 'price 2022', 'price 2023']]  # Historical prices for regression
y_future = df['price']  # Current prices

# Initialize and train the linear regression model for future price prediction
future_price_model = LinearRegression()
future_price_model.fit(X_future, y_future)

def get_unique_locations():
    df = pd.read_csv('bengaluru_house_prices.csv')  # Adjust the file path if needed
    locations = df['location'].dropna().unique()
    locations = [loc for loc in locations if isinstance(loc, str)]  # Keep only strings
    return sorted(locations)  # Sort locations alphabetically

# Prediction function as per your Jupyter code
def predict_price(location, sqft, bath, bhk):
    # Create a zero array for input features
    x = np.zeros(len(feature_columns))

    # Fill in the relevant features
    x[feature_columns.index('total_sqft')] = sqft
    x[feature_columns.index('bath')] = bath
    x[feature_columns.index('bhk')] = bhk
    loc_index = feature_columns.index(location) if location in feature_columns else -1
    if loc_index >= 0:
        x[loc_index] = 1  # Set location feature

    # Predict current price using the trained model
    return model.predict([x])[0]

@app.route('/')
def index():
    locations = get_unique_locations()
    return render_template('homepage.html', locations=locations)

@app.route('/login')
def login():
    return render_template('login.html')

def increase_price(predicted_price, percentage_increase=random.randint(5, 10)):
    # Calculate the new price by increasing the predicted price
    new_price = predicted_price * (1 + percentage_increase / 100)
    return new_price

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/propworth', methods=['POST'])
def predict():
    # Get the input value from the form
    locality = request.form['locality']
    return render_template('propworth.html', locality=locality)

def format_price(price):
    if price >= 100:  # Treat any 3-digit number or higher as crore
        return f"{price / 1e2:.2f} Crore"
    else:
        return f"{price:.2f} Lakhs"

@app.route('/prediction', methods=['POST'])
def prediction():
    # Get input values from the form
    location = request.form.get('projectLocality')  # Categorical (string)
    super_area = float(request.form.get('superArea'))  # Convert to float (numeric)
    bath = int(request.form.get('bath'))  # Convert to int (numeric)
    bhk = int(request.form.get('bhk'))  # Convert to int (numeric)

    # Predict current price using the location, area, bath, and bhk
    predicted_price = predict_price(location, super_area, bath, bhk)
    formatted_price = format_price(predicted_price)

    # Calculate future price
    future_price = increase_price(predicted_price)

    # Store predicted price and future price in session or pass to the next route
    return render_template('prediction.html',
                           project_locality=location,
                           super_area=super_area,
                           bath=bath,
                           bhk=bhk,
                           predicted_price=formatted_price,
                           future_price=future_price)  # Redirect to future prediction instead

@app.route('/estimate_input', methods=['POST'])
def estimate():
    locality = request.form['locality']  # Get the selected locality from the form
    # You can add additional logic here if needed

    return render_template('estimate_input.html', locality=locality)


@app.route('/estimate_future', methods=['POST'])
def future_estimate():
    project_locality = request.form['projectLocality']
    property_type = request.form['propertyType']
    bhk = request.form['bhk']
    bath = request.form['bath']
    super_area = request.form['superArea']

    print(request.form)  # This will print all form data to check if the key is present

    # Predict current price using the location, area, bath, and bhk
    predicted_price = predict_price(project_locality, float(super_area), int(bath), int(bhk))
    future_price = format_price(increase_price(predicted_price))

    return render_template('future_estimate.html',
                           project_locality=project_locality,
                           property_type=property_type,
                           bhk=bhk,
                           super_area=super_area,
                           future_price=future_price)



if __name__ == "__main__":
    app.run(debug=True, port=5001)
