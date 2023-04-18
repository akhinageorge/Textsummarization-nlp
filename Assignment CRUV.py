import requests
import re
from io import StringIO
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') 
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords 

#Step1: Data acquisition
url='https://drive.google.com/file/d/1KsWWrhGaK0_IMXPaYBOEXXE8IExqogmJ/edit'

file_id = url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url2 = requests.get(dwn_url).text
csv_raw = StringIO(url2)
df = pd.read_csv(csv_raw)

#Step2: Text Preprocessing
def preprocess_text_wf(corpus):
    corpus = re.sub(r'\[[0-9]*\]', ' ', corpus)     
    corpus = re.sub(r'\s+', ' ', corpus) 
    formatted_corpus = re.sub(r'[^a-zA-Z]', ' ', corpus)  
    formatted_corpus = re.sub(r'\s+', ' ', formatted_corpus) 
    
    sentences_wf = nltk.sent_tokenize(corpus)        
    return formatted_corpus, sentences_wf

def remove_stopwords(sent):
    stopWords = nltk.corpus.stopwords.words('english')
    new_sent = " ".join([word for word in sent if word not in stopWords])
    return new_sent

#Step3: Featured Engineering 
def build_freqDist(corpus):
    words = nltk.word_tokenize(corpus)
    word_freqs = nltk.FreqDist(words)
    max_freq = max(word_freqs.values())
    word_freqs_normalized = {k:v/max_freq for k,v in word_freqs.items()}
      
    return word_freqs_normalized

def calculate_sent_scores(wordFreqs, sentence_list):
    sent_scores = {}

    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in wordFreqs.keys():
                if len(sent.split()) < 35:   
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = wordFreqs[word]
                    else:
                        sent_scores[sent] += wordFreqs[word]
    
    return sent_scores

def generate_summary_wordFreq(sentScores, numSents):
    sorted_scores = sorted(sentScores.items(), key=lambda x: x[1], reverse=True)
    sents_scored = [k for k,v in sorted_scores]
    summary = ' '.join(sents_scored[:numSents])
    return summary

#Step4: Modelling

num_sents=5
para=str(df.iloc[67,1])
formatted_corpus_wf, sentences_wf = preprocess_text_wf(para)
sentences_wf_no_sw = remove_stopwords(nltk.word_tokenize(formatted_corpus_wf))      
weighted_wordFreqs = build_freqDist(sentences_wf_no_sw)                             
sent_scores = calculate_sent_scores(weighted_wordFreqs, sentences_wf)               
summary_word_freq = generate_summary_wordFreq(sent_scores, num_sents)               

#Step5: Model Evaluation 
print("\n Original Text \n", para)
sentenceso = nltk.word_tokenize(para)
print("\n No of Words in Original Text \n", len(sentenceso))
print("\n Summary \n", summary_word_freq)
sentencess = nltk.word_tokenize(summary_word_freq)
print("\n No of Words in Original Text \n", len(sentencess))

