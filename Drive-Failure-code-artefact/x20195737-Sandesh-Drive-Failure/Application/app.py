from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('CONV_GRU.h5')

# Define all column names
columns = [
    "smart_196_parameter", "smart_5_parameter", "smart_198_parameter",
    "smart_184_parameter", "smart_187_parameter", "smart_2_parameter",
    "smart_7_parameter", "smart_199_parameter", "smart_24_parameter",
    "capacity_bytes", "smart_241_parameter", "smart_193_parameter",
    "smart_191_parameter", "smart_1_parameter", "smart_197_parameter"
]

# Define default values for the specific columns
default_values_dict = {
    "smart_2_parameter": 0.0,
    "smart_7_parameter": 0.3669940381665169,
    "smart_199_parameter": 1.0,
    "smart_24_parameter": 0.0,
    "capacity_bytes": 0.44025225181193633,
    "smart_241_parameter": 0.3952569169960474,
    "smart_193_parameter": 0.32241248808981127,
    "smart_191_parameter": 0.9482749920598742,
    "smart_1_parameter": 0.3970624404490563,
    "smart_197_parameter": 0.3968253968253968
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_message = None
    if request.method == 'POST':
        # Retrieve values for the first five features from the form
        input_values = {column: float(request.form.get(column, 0.0)) for column in columns[:5]}

        # Combine user input with default values for the remaining features
        combined_values = [input_values.get(column, default_values_dict.get(column, 0.0)) for column in columns]

        # Reshape the data for the model
        data_test = np.array([combined_values]).reshape((1, 1, 1, 15))

        # Make a prediction
        prediction = model.predict(data_test)

        # Use a threshold of 0.5 to determine the class label
        disk_failure_risk = (prediction[0] > 0.5).astype(int)

        # Create a message based on the prediction outcome
        if disk_failure_risk:
            prediction_message = 'Disk may fail soon. Please back up your data.'
        else:
            prediction_message = 'Disk failure chances are very rare.'

    # Render the index page with the prediction result
    return render_template('index.html', prediction_message=prediction_message, input_columns=columns[:5], default_columns=list(default_values_dict.keys()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
