# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:28:23 2021

@author: makn0023
"""
from flask import Flask, render_template, request, redirect, url_for
from flask_restful import Api
#import request
import requests
from flask import *
import json
import pandas as pd
from model import dataclean, dataft, ad

dfj = dataclean.copy()
dfjt = dataft.copy()
html=[]
app = Flask(__name__)
#api = Api(app)
@app.route('/genres')
def home():
    return render_template('index.html')

@app.route('/genres/train')
def train():
    #data = request.get_json(force=True)
    result = dfj.to_json(orient="table")
    parsed = json.loads(result)
    #load= json.dumps(parsed, indent=4)  
    #load = json.dumps(dfj.to_json(orient='table'))
    #html = list(dfj.groupby("genres"))
    #html = dfj.to_html()
    return parsed

@app.route('/genres/predict')
def predict():
    result = ad.to_json(orient="table")
    parsed = json.loads(result)
    ab = ad.to_csv('predicted.csv')
    #data = request.get_json(force=True)
    #load = json.dumps(dfj.to_json(orient='table'))
    #html = dfjt.to_html()
    return parsed, ab

if __name__ == '__main__':
    app.debug = True
    app.run()
#if __name__ == "__main__":
#    app.run(debug=True)