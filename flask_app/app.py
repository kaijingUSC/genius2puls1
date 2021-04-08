#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask import Flask, render_template
from flask import url_for, escape, request, redirect, flash
from flask_bootstrap import Bootstrap
from util import search_stock


app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    news=[]
    if request.method == 'POST':
        key = request.form.get('symbol')
        # print(key)
        if not key:
            flash('Enter a symbol to start')
            return redirect(url_for('index'))
        news = search_stock(key)
    print(news)
    return render_template('index.html', news=news)

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

name = 'Grey Li'
movies = [
    {'title': 'My Neighbor Totoro', 'year': '1988'},
    {'title': 'Dead Poets Society', 'year': '1989'},
    {'title': 'A Perfect World', 'year': '1993'},
    {'title': 'Leon', 'year': '1994'},
    {'title': 'Mahjong', 'year': '1996'},
    {'title': 'Swallowtail Butterfly', 'year': '1996'},
    {'title': 'King of Comedy', 'year': '1999'},
    {'title': 'Devils on the Doorstep', 'year': '1999'},
    {'title': 'WALL-E', 'year': '2008'},
    {'title': 'The Pork of Music', 'year': '2012'},
]

