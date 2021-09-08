

import flask
from flask import Flask, request, render_template
from flask_cors import CORS
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from newspaper import Article
import urllib
import joblib

# load the model from disk
file_name = 'nlp_model.pkl'
fnd = pickle.load(open(file_name, 'rb'))
cv = pickle.load(open('transformation.pkl','rb'))
app = Flask(__name__)
    

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    data = [news]
    vect = cv.transform(data).toarray()
    my_pred = fnd.predict(vect)
    return render_template('result.html', prediction = my_pred)

if __name__=="__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
    