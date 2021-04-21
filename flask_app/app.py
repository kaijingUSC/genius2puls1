#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask import Flask, render_template
from flask import url_for, escape, request, redirect, flash
from flask_bootstrap import Bootstrap
from util import search_stock, stock_predict, moving_average, Rate_of_Return, Correlation, Risk_and_Return, get_plt


app = Flask(__name__)
bootstrap = Bootstrap(app)

key = True
predict_df = []
historical_df = []
news = []
prediction = []
symbol = ''
company = ''

@app.route('/', methods=['GET', 'POST'])
def index():
    # news=[]
    img_url=False
    # prediction=[]
    # symbol=''
    # company=''
    timeRange = 30
    if request.method == 'POST':
        # global key
        # key = request.form.get('symbol')
        # print(key)
        # if not key:
        #     flash('Enter a symbol to start')
        #     return redirect(url_for('index'))
        if request.form.get('symbol'):
            global key 
            global predict_df
            global historical_df
            global news
            global prediction
            global symbol
            global company

            key = request.form.get('symbol')
            print(key)
            symbol, company = search_stock(key)
            predict_df, historical_df, news, prediction= stock_predict(symbol, company)
        
        elif request.form['range'] :
            print(request.form['range'])
            if request.form['range'] == "Three Months":
                timeRange = 30
            elif request.form['range'] == "Half Year":
                timeRange = 180
            elif request.form['range'] == "One Year":
                timeRange = len(historical_df)
    # print(predict_df)
    if len(predict_df) > 0:
        img_url = get_plt(predict_df, historical_df, timeRange)
    return render_template('index.html', symbol=symbol, company=company, news=news, img_url=img_url, prediction=prediction)


@app.route('/other_analysis', methods=['GET', 'POST'])
def other_analysis():
    img_url=''
    if request.method == "POST":
        if request.form['link'] == 'Moving Average':
            if not key:
                flash('Choose one')
                return redirect(url_for('other_analysis'))
            img_url = moving_average(key)
        elif request.form['link'] == 'Rate of Return':
            if not key:
                flash('Choose one')
                return redirect(url_for('other_analysis'))
            img_url = Rate_of_Return(key)
        elif request.form['link'] == 'Correlation':
            if not key:
                flash('Choose one')
                return redirect(url_for('other_analysis'))
            img_url = Correlation(key)
        elif request.form['link'] == 'Risk and Return':
            if not key:
                flash('Choose one')
                return redirect(url_for('other_analysis'))
            img_url = Risk_and_Return(key)
    return render_template('other_analysis.html', img_url=img_url)

@app.route('/contact_us', methods=['GET', 'POST'])
def contact_us():
    return render_template('contact_us.html')

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


