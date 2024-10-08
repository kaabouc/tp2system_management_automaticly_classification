from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import io
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

models = {
    "logistic_regression": None,
    "random_forest": None,
    "knn": None
}

@app.route('/upload', methods=['POST'])
def upload_dataset():
    file = request.files['file']
    df = pd.read_csv(file)
    return jsonify({"columns": df.columns.tolist()})

@app.route('/train', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the file content
    file_content = file.read()
    
    # Get JSON data
    feature_columns = request.form.get('featureColumns')
    target_column = request.form.get('targetColumn')

    if not feature_columns or not target_column:
        return jsonify({"error": "Missing feature columns or target column"}), 400

    feature_columns = eval(feature_columns)  # Convert string representation to list
    
    # Read the CSV file
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    algorithms = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'knn': KNeighborsClassifier()
    }
    
    for name, model in algorithms.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        models[name] = model
        joblib.dump(model, f"models/{name}.pkl")
    
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        algorithm = data['algorithm']
        input_data = data['input']
        
        print(f"Received prediction request for algorithm: {algorithm}")
        print(f"Input data: {input_data}")
        
        model = models.get(algorithm)
        if not model:
            return jsonify({"error": f"Model '{algorithm}' not trained yet!"}), 400

        # Convert input data to float
        input_data = [float(x) for x in input_data]
        
        # Reshape input data to 2D array
        input_array = np.array(input_data).reshape(1, -1)
        
        prediction = model.predict(input_array)
        print(f"Prediction result: {prediction}")
        
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)