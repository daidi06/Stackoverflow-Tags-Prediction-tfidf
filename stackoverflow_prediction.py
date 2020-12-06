#! /usr/bin/env python3
# coding: utf-8

from model import PredictTags
import os
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.widgets import TextArea

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

predicter = PredictTags()

class ReusableForm(Form):
    name = TextAreaField('Post:',
                        validators=[validators.required()],
                        widget=TextArea(),
                        render_kw={"rows": 20, "cols": 120})
    
@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
    
    print(form.errors)
    
    if request.method == 'POST':
        name=request.form['name']
        print(name)
    
    if form.validate():
        tags = ' | '.join(predicter.text2tags(name))
        flash(tags)
    else:
        flash('')
    
    return render_template('layout.html', form=form)

if __name__ == "__main__":
    app.run()