from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('RFRegressorModel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

order = ['carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
       'horsepower', 'citympg', 'highwaympg', 'fueltype_diesel',
       'fueltype_gas', 'enginelocation_front', 'enginelocation_rear',
       'aspiration_std', 'aspiration_turbo', 'doornumber_four',
       'doornumber_two', 'carbody_convertible', 'carbody_hardtop',
       'carbody_hatchback', 'carbody_sedan', 'carbody_wagon', 'drivewheel_4wd',
       'drivewheel_fwd', 'drivewheel_rwd', 'enginetype_dohc',
       'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf',
       'enginetype_ohcv', 'enginetype_rotor', 'cylindernumber_eight',
       'cylindernumber_five', 'cylindernumber_four', 'cylindernumber_six',
       'cylindernumber_three', 'cylindernumber_twelve', 'cylindernumber_two',
       'fuelsystem_1bbl', 'fuelsystem_2bbl', 'fuelsystem_4bbl',
       'fuelsystem_idi', 'fuelsystem_mfi', 'fuelsystem_mpfi',
       'fuelsystem_spdi', 'fuelsystem_spfi']

# Define the headers for one-hot encoding
headers = ['fueltype', 'enginelocation', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
           'enginetype', 'cylindernumber', 'fuelsystem']
numerical_columns = ['carlength','carwidth','carheight','curbweight','enginesize','horsepower','citympg','highwaympg']
all_col = ['carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
       'horsepower', 'citympg', 'highwaympg', 'fueltype_diesel',
       'fueltype_gas', 'enginelocation_front', 'enginelocation_rear',
       'aspiration_std', 'aspiration_turbo', 'doornumber_four',
       'doornumber_two', 'carbody_convertible', 'carbody_hardtop',
       'carbody_hatchback', 'carbody_sedan', 'carbody_wagon', 'drivewheel_4wd',
       'drivewheel_fwd', 'drivewheel_rwd', 'enginetype_dohc',
       'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf',
       'enginetype_ohcv', 'enginetype_rotor', 'cylindernumber_eight',
       'cylindernumber_five', 'cylindernumber_four', 'cylindernumber_six',
       'cylindernumber_three', 'cylindernumber_twelve', 'cylindernumber_two',
       'fuelsystem_1bbl', 'fuelsystem_2bbl', 'fuelsystem_4bbl',
       'fuelsystem_idi', 'fuelsystem_mfi', 'fuelsystem_mpfi',
       'fuelsystem_spdi', 'fuelsystem_spfi']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect user input from the form
        user_input = {}
        for header in headers:
            user_input[header] = request.form[header]

        user_input['carlength'] = (float(request.form['carlength'])*39.3701)
        user_input['carwidth'] = (float(request.form['carwidth'])*39.3701)
        user_input['carheight'] = (float(request.form['carheight'])*39.3701)
        user_input['curbweight'] = (float(request.form['curbweight'])*2.20462)
        user_input['enginesize'] = (float(request.form['enginesize'])*0.0610237)
        user_input['horsepower'] = int(math.floor(float(request.form['horsepower'])))
        user_input['citympg'] = (int(math.floor(float(request.form['citympg'])*0.621371)))
        user_input['highwaympg'] = (int(math.floor(float(request.form['highwaympg'])*0.621371)))
        user_input['fueltype'] = request.form['fueltype']
        user_input['enginelocation'] = request.form['enginelocation']
        user_input['aspiration'] = request.form['aspiration']
        user_input['doornumber'] = request.form['doornumber']
        user_input['carbody'] = request.form['carbody']
        user_input['drivewheel'] = request.form['drivewheel']
        user_input['enginetype'] = request.form['enginetype']
        user_input['cylindernumber'] = request.form['cylindernumber']
        user_input['fuelsystem'] = request.form['fuelsystem']
        # Convert user input to a DataFrame
        user_input_df = pd.DataFrame([user_input])

        # One-hot encode categorical variables
        user_input_df = pd.get_dummies(columns=headers, data=user_input_df)
        for i in all_col: 
            if i not in user_input_df.columns:
                user_input_df[i]=False

        # Apply feature scalinga
        user_input_df[numerical_columns] = scaler.transform(user_input_df[numerical_columns])
        user_input_df=user_input_df[order]
        # Make prediction
        prediction = model.predict(user_input_df)[0]*83.38
        return render_template('index.html', prediction_text=f'₹{prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
