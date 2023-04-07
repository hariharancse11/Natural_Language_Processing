#text = 'Nigel Farage, head of the UK Independence Party and a leading voice in favor of leaving the EU, told Sky News he did not expect to be on the winning side.'

import sklearn

import joblib
# load the model from disk
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

CLASS_NAMES = ['BUSINESS',
 'ENTERTAINMENT',
 'FOOD & DRINK',
 'PARENTING',
 'POLITICS',
 'SPORTS',
 'STYLE & BEAUTY',
 'TRAVEL',
 'WELLNESS',
 'WORLD NEWS']

import regex as re
def remove(text):
  #remove mention
  text = re.sub("@[A-Za-z0-9_]+","", text)
  # remove stock market tickers like $GE
  text = re.sub(r'\$\w*', '', text)
  # remove old style retext text "RT"
  text = re.sub(r'^RT[\s]+', '', text)
  text = re.sub(r'^rt[\s]+', '', text)
  # remove hyperlinks
  text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
  text = re.sub(r'^https[\s]+', '', text)
  # remove hashtags
  # only removing the hash # sign from the word
  text = re.sub(r'#', '', text)
  text = re.sub(r'%', '', text)
  #remove coma
  text = re.sub(r',','',text)
  #remove angka
  text = re.sub('[0-9]+', '', text)
  text = re.sub(r':', '', text)
  #remove space
  text = text.strip()
  #remove double space
  text = re.sub('\s+',' ',text)
  return text

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#clean stopwords
stopword = set(stopwords.words('english'))
def clean_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in stopword) 
    return text


from nltk.stem import PorterStemmer
ps = PorterStemmer()
def porterstemmer(text):
  text = ' '.join([ps.stem(word) for word in text.split() if word in text])
  return text

import spacy
nlp = spacy.load('en_core_web_sm')

def lemmatization (text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

def autoCategorize(txt):
  res = remove(txt.lower())
  res = clean_stopwords(res)
  res = porterstemmer(res)
  res = lemmatization(res)
  vec = vectorizer.transform([res])

  return(CLASS_NAMES[loaded_model.predict(vec)[0]])
