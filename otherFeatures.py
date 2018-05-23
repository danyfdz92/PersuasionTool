import numpy
import re
import csv
from sklearn.externals import joblib
from textstat.textstat import textstat
import nltk
import collections as ct
from nltk import word_tokenize

def gettingFeatures(text):
        text = text.lower()
	
	#words / syllables / sentences count
        wordCount = len(text.split())
        syllables = textstat.syllable_count(text)
        sentences = textstat.sentence_count(text)
        try:
            #ReadabilityScore
            readabilityScore = 206.835 - 1.015 * (wordCount / sentences) - 84.6 * (syllables / wordCount);
            #ReadabilityGrade
            ReadabilityGrade = 0.39 * (wordCount / sentences) + 11.8 * (syllables / wordCount) - 15.59;
        except:
            readabilityScore = 0
            ReadabilityGrade = 0
        print(readabilityScore,ReadabilityGrade)
        #Direction Count
        #private String[] direction = {"here", "there", "over there", "beyond", "nearly", "opposite", "under", "above", "to the left", "to the right", "in the distance"};
        DiractionCount  = 0
        DiractionCount = text.count("here") + text.count("there") + text.count("over there") + text.count("beyond") + text.count("nearly") + text.count("opposite") + text.count("under") + text.count("to the left") + text.count("to the right") + text.count("in the distance") 
        #Exemplify count
	#private String[] exemplify = {"chiefly", "especially", "for instance", "in particular", "markedly", "namely", "particularly", "including", "specifically", "such as"};
        Exemplify = 0
        Exemplify = text.count("chiefly") + text.count("especially") + text.count("for instance") + text.count("in particular") + text.count("markedly") + text.count("namely") + text.count("particularly")+ text.count("incluiding") + text.count("specifically") + text.count("such as") 
        
        try:
            #words per sentence (average)
            WPS = 0
            parts = [len(l.split()) for l in re.split(r'[?!.]', text) if l.strip()]
            WPS = sum(parts)/len(parts) #number of words per sentence
        except:
            WPS = 0
        #print(wordCount, readabilityScore, ReadabilityGrade, DiractionCount, WPS, Exemplify)
        return numpy.array([wordCount, readabilityScore, ReadabilityGrade, DiractionCount, WPS, Exemplify])

results = numpy.array([0, 0, 0, 0, 0, 0])
with open('persuasive_originalPost.csv', newline='', encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for row in reader:
        comment = row[1]
        features = gettingFeatures(comment)
        results = numpy.vstack((results,features))
"""
import pandas as pd 
df = pd.DataFrame(results)
df.to_csv("nonpersuasive_features.csv")"""