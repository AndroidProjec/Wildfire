from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('wildfire_rnn.h5')

# Load your actual training dataset used for fitting the scaler during training
training_data = pd.read_csv('C:/Users/saeed/Documents/Ansar/DeepLearning/FinalSubmission/dataset.csv')

# Extract numerical columns used for scaling
numerical_columns = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']

# Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(training_data[numerical_columns])

@app.route('/')
def home():
    return render_template('index.html', prediction=None)
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_data = [float(request.form.get(f)) for f in ['latitude', 'longitude', 'brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']]

    # Ensure the input data has the same number of features as expected by the model (including 'daynight')
    if len(input_data) != 9:
        return render_template('index.html', prediction="Error: Invalid number of features")

    # Add the 'daynight' feature (assumed to be a boolean value)
    input_data.append(request.form.get('daynight') == 'TRUE')

    # Preprocess the user input
    input_features = scaler.transform(pd.DataFrame([input_data]))

    # Reshape the input to match the expected shape (assuming LSTM input shape)
    input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

    # Make predictions using the loaded model
    predictions = model.predict(input_features)

    # Return the predictions
    return render_template('index.html', prediction=predictions[0][0])

if __name__ == '__main__':
    app.run(debug=True)
