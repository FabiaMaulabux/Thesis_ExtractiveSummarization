import numpy as np
import random
import json
import re
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def amiModel():
    model = load_model(r'C:\Users\FMaulabux\Documents\Thesis\Defence\flaskapp\models\model_mlp.h5')
    word_index = json.load(open(r'C:\Users\FMaulabux\Documents\Thesis\Defence\flaskapp\data\vocab.json'))
    print("Loaded model and vocabulary")
    return model, word_index

def articleModel():
    model = load_model(r'C:\Users\FMaulabux\Documents\Thesis\Defence\flaskapp\models\model_cnn2.h5')
    word_index = json.load(open(r'C:\Users\FMaulabux\Documents\Thesis\Defence\flaskapp\data\vocab_article.json'))
    print("Loaded model and vocabulary")
    return model, word_index

def BOW(sentence):
    WPT = nltk.WordPunctTokenizer()
    #nltk.download('stopwords')
    #nltk.download('punkt')
    #nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_word_list = nltk.corpus.stopwords.words('english')
    '''Remove numbers and special characters in sentence'''
    sentence = re.sub(" \d+", " ", sentence) #digits
    sentence = re.sub(r'[0-9]+', "",sentence) #digits
    sentence = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", sentence)
    pattern = r"[{}]".format("-_)(,;:$%#.") #special characters
    sentence = re.sub(pattern, "", sentence) 
    sentence = re.sub(r"[\']", "",sentence)
    sentence = re.sub(r"[/']", "",sentence)
    sentence = re.sub(r'\b[a-zA-Z]\b', '', sentence) #remove single letter words
    sentence = re.sub('\s+', ' ', sentence).strip() #remove double spaces
    '''Lowercase'''
    sentence = sentence.lower()
    sentence = sentence.strip()
    '''Tokenize'''
    tokens = WPT.tokenize(sentence)
    filtered_tokens = [token for token in tokens if token not in stop_word_list]
    
    '''Lem'''
    k = []
    for word in range(len(filtered_tokens)):
        k.append(lemmatizer.lemmatize(filtered_tokens[word]))
    sentence = ' '.join(k)
    return sentence

# Generate output
def toArray(sentence, word_index):
    
    sentence_tokenized = nltk.word_tokenize(sentence)
    x = [word_index.get(word, 0)
                  for word in sentence_tokenized]
    return x

def padsequence(sequences, sequencelength):
    sequences = pad_sequences(sequences, padding='post', maxlen=sequencelength)
    return sequences

def generateLabels(predictions, threshold):
    scores = ['none'] * len(predictions)
    labels = [0] * len(predictions)
    for i in range(len(predictions)):
        scores[i] = predictions[i][1]
        if scores[i] > threshold:
            labels[i] = 1
    return labels

def generate_from_seed(model, word_index, seed, threshold):
    """Generate output from a sequence"""
    
    sentence_text = nltk.sent_tokenize(seed)
    sentence_text_original = sentence_text
    
    for i in range(len(sentence_text)):
        sentence_text[i] = BOW(sentence_text[i])
    print("Preprocessed all " + str(len(sentence_text)) + " sentences")    

    
    arrayList = [toArray(item, word_index) for item in sentence_text]
    print(arrayList)
    
    arrayList = padsequence(arrayList, 100)
    print("Converted to numeric padded sequences")
    
    sentence_text_original = nltk.sent_tokenize(seed)
    
    df = pd.DataFrame(sentence_text_original)
    df.columns = ["Text"]
        
    # Make predictions
    predictions = model.predict(arrayList)
    print("Predicted sentences")
    #Create transcript based on predictions and threshold
   
    threshold = float(threshold)
    
    labels = generateLabels(predictions, threshold)
    print(labels)
    df['Prediction'] = labels
    transcript = df.Text[df.Prediction == 1]
  
  #Concatenate all transcript sentences
    print("Generating summary")
    transcript = pd.DataFrame(transcript)
    t = ""
  
    for i in range(len(transcript)):
        t = t + str(transcript.iloc[i,0]) + " "
  

    # Formatting in html
    html = ''
    html = addContent(html, header(
        'GENERATED SUMMARY', color='black', gen_text=''))
    html = addContent(html, box(t))
    return '<div>{}</div>'.format(html)


def header(text, color='black', gen_text=None):
    """Create an HTML header"""


    raw_html = '<h1 style="margin-top:12px;color: {};font-size:24px"><center>'.format(color) + str(
		text) + '</center></h1>'
    return raw_html


def box(text, gen_text=None):
    """Create an HTML box of text"""

    if gen_text:
        raw_html = '<div style="padding:8px;font-size:14px;margin-top:28px;margin-bottom:14px;">' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</div>'

    else:
        raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 14px;">' + str(
            text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html