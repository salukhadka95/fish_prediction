from flask import Flask, render_template, request

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)


# Load the trained model
# Load the model from the file
with open('prediction.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])
        species = request.form['species']

        # Convert species to one-hot encoded format
        species_cols = ['Species_Bream', 'Species_Parkki', 'Species_Perch', 'Species_Pike', 'Species_Roach', 'Species_Smelt', 'Species_Whitefish']
        species_encoded = [1 if species == col.split('_')[-1] else 0 for col in species_cols]

        # Create input data in DataFrame format
        input_data = pd.DataFrame({
            'Length1': [length1],
            'Length2': [length2],
            'Length3': [length3],
            'Height': [height],
            'Width': [width],
            **{col: [val] for col, val in zip(species_cols, species_encoded)}
        })

        # Make prediction
        predicted_weight = model.predict(input_data)[0]

        return render_template('index.html', prediction=predicted_weight)
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
