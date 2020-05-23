import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RCF.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    #prediction = model.predict(final_features)[0]
    prediction = "Congratulations !!! Your order won't be Returned." if model.predict(final_features)[0]==0 else "I'm sorry to say this. But there is a great chance that your Order will be Returned."

    return render_template('index.html', prediction_text='{}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
