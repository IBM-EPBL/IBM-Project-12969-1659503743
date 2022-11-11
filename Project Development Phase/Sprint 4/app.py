from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import joblib
scaler = joblib.load("my_scaler.save")


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/home")
@app.route("/")
def hello():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        feature_names = ["ph", "Hardness" , "Solids", "Chloramines", "Sulfate",
                         "Conductivity", "Organic_carbon","Trihalomethanes", "Turbidity"]

        df = pd.DataFrame(features_value, columns = feature_names)
        df = scaler.transform(df)
        output = model.predict(df)

        if output[0] == 1:           
            prediction = "safe"
        else:
            prediction = "not safe"


        return render_template('home.html', prediction_text= "water is {} for human consumption ".format(prediction))

        

if __name__ == "__main__":
    app.run(debug=True)
