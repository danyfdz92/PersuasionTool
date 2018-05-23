import csv
import re
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0,1]
    except:
        return 0


#print(cosine_sim('a little bird', 'a little bird'))
results_tfidf = []
with open('persuasive_originalPost.csv', newline='', encoding="ISO-8859-1") as f:
    reader = csv.reader(f)
    for row in reader:
        comment = row[1]
        original_comment = row[6]
        print(cosine_sim(comment, original_comment))
        results_tfidf.append(cosine_sim(comment, original_comment))

with open("output_Persuasive.csv",'w') as f:
    wr = csv.writer(f)
    wr.writerows(map(lambda x: [x], results_tfidf))