# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:04:03 2023

@author: david redondo
based in a example from https://realpython.com/python-web-applications/#improve-the-user-interface-of-your-web-application
"""
from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

@app.route("/")
def index():
    return "Congratulations, it's a web app!"

@app.route("/<int:celsius>")
def fahrenheit_from(celsius):
    """Convert Celsius to Fahrenheit degrees."""
    fahrenheit = float(celsius) * 9 / 5 + 32
    fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
    return str(fahrenheit)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)