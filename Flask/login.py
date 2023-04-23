# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:13:47 2023

@author: david
"""

from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
 
 
@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name
 
 
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))
 
 
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)