import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('/Users/mandeebot/Desktop/ml_classification/wep-app/data_set/rf_model.pkl','rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])

def predict():
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    case_status = ["Certified","Denied"]

    return render_template(
        "index.html", prediction_text="Likely visa status: {}".format(case_status[output])
    )

@app.route('/results',methods=['POST'])

def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)