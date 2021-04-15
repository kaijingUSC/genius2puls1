#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask import Flask, render_template
from flask import url_for, escape, request, redirect, flash
from flask_bootstrap import Bootstrap
from util import search_stock, stock_predict_plt


app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    news=[]
    img_url=''
    if request.method == 'POST':
        key = request.form.get('symbol')
        # print(key)
        if not key:
            flash('Enter a symbol to start')
            return redirect(url_for('index'))
        news = search_stock(key)
        img_url = stock_predict_plt(key)
    return render_template('index.html', news=news, img_url=img_url)


@app.route('/other_analysis', methods=['GET', 'POST'])
def other_analysis():
    return render_template('other_analysis.html')


@app.route('/user/<name>')
def user_page(name):
    return 'User: %s' % escape(name)

@app.route('/test')
def test_url_for():
    print(url_for('user_page', name='greyli')) 
    print(url_for('user_page', name='peter'))
    print(url_for('test_url_for'))
    print(url_for('test_url_for', num=2))
    return 'Test Page'


