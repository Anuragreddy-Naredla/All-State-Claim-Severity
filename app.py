from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np
import flask
import xgboost as xgb1

model= pickle.load(open("XGBOOST_FINAL_SUBMISSION_MODEL.pkl", "rb"))

app = Flask(__name__)
          
    

@app.route('/')
def hello_world():
    return 'Hello World!'
    
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.files['file']
        data = pd.read_csv(f.filename,sep="\t")
        d_test =  xgb1.DMatrix(data)
        pred = np.exp(model.predict(d_test)) - 100
        result = dict(enumerate(pred.flatten(), 1))
        return render_template("success.html", name = result)
    except:
        return render_template("error.html")

if __name__ == '__main__':
    app.debug = True
    app.run()