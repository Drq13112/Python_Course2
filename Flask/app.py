# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:20:50 2023

@author: David Redondo Quintero
"""


from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World!"

@app.route('/hi/')
def who():
    return "Hello World!"

@app.route('/hi/<username>')
def greet(username):
    return f"Hi there, {username}!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)