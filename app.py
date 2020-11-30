# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:17:23 2020

@author: Sushilkumar
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_suv.pkl', 'rb'))
modelDbs=pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
   # isPurchased=model.predict([[age,salary]])
    output = round(prediction[0], 2)
    print(output)
    if output==0:
        print("User not Intersted to Buy SUV")
        prediction_text="User not Intersted to Buy SUV"
    elif output==1:
        print("User Intersted to Buy SUV")
        prediction_text="User Intersted to Buy SUV"
 
    return render_template('index.html', prediction_text=prediction_text)


@app.route('/predictDb',methods=['POST'])
def predictDb():
    int_features = [[int(x) for x in request.form.values()]]
    print("diabetes",int_features)
    predictiondb = modelDbs.predict(int_features)
    print("predictiondb final",predictiondb[0])
    outputdb=predictiondb[0]
    print(outputdb)
    if outputdb==0:
        print("Congratulations!!!... You are not suffering from Diabetes")
        prediction_text_db="Congratulations!!!... You are not suffering from Diabetes"
    elif outputdb==1:
        print("OOps!!!... You are suffering from Diabetes. Plzz visit nearest doctor")
        prediction_text_db="OOps!!!... You suffering from Diabetes. Plzz visit nearest doctor"
    
    return render_template('index.html', predictiondb=prediction_text_db)
    



if __name__ == "__main__":
    app.run(debug=False)