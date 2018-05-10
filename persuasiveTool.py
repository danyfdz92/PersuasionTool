from appJar import gui
import numpy
import re
from sklearn.externals import joblib
from textstat.textstat import textstat
import nltk
import collections as ct
from nltk import word_tokenize

#filename = 'finalized2_model.sav'
#loaded_model = joblib.load(filename)

app=gui()

def press(btn):
        if btn == "Cancel":
        	app.stop()
        else:
            content = app.getTextArea("Add text here")
            selectedModels = app.getAllCheckBoxes()
            results = [key for key, val in selectedModels.items() if val == True]
            data = gettingFeatures(content)
            if bool(results):
                result = ""
                for model in results:
                    if model == "Logistic Regression":
                        loaded_model = joblib.load("LogisticRegression.sav")
                        predictions = loaded_model.predict(data.reshape(1,-1))
                        if predictions[0] < 1:
                            result += "\n\n is Non persuasive using " + model
                        else :
                            result += "\n\n is Persuasive using " + model
                    if model == "Linear Discriminant Analysis":
                        loaded_model = joblib.load("LinearDiscriminantAnalysis.sav")
                        predictions = loaded_model.predict(data.reshape(1,-1))
                        if predictions[0] < 1:
                            result += "\n\n is Non persuasive using " + model
                        else :
                            result += "\n\n is Persuasive using " + model
                    if model == "KNeighbors Classifier":
                        loaded_model = joblib.load("KNeighborsClassifier.sav")
                        predictions = loaded_model.predict(data.reshape(1,-1))
                        if predictions[0] < 1:
                            result += "\n\n is Non persuasive using " + model
                        else :
                            result += "\n\n is Persuasive using " + model
                    if model == "Decision Tree Classifier":
                        loaded_model = joblib.load("DecisionTreeClassifier.sav")
                        predictions = loaded_model.predict(data.reshape(1,-1))
                        if predictions[0] < 1:
                            result += "\n\n is Non persuasive using " + model
                        else :
                            result += "\n\n is Persuasive using " + model
                    if model == "Gaussian Naive Bayes":
                        loaded_model = joblib.load("GaussianNB.sav")
                        predictions = loaded_model.predict(data.reshape(1,-1))
                        if predictions[0] < 1:
                            result += "\n\n is Non persuasive using " + model
                        else :
                            result += "\n\n is Persuasive using " + model
                    if model == "Support Vector Machine":
                        loaded_model = joblib.load("SupportVectorMachine.sav")
                        predictions = loaded_model.predict(data.reshape(1,-1))
                        if predictions[0] < 1:
                            result += "\n\n is Non persuasive using " + model
                        else :
                            result += "\n\n is Persuasive using " + model
                result = "The text: \n"+ content + result
            else:
                result = "You need to choose at least one classification model"
            
            app.infoBox("Success", result)

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
        #Direction Count
        #private String[] direction = {"here", "there", "over there", "beyond", "nearly", "opposite", "under", "above", "to the left", "to the right", "in the distance"};
        DiractionCount  = 0
        DiractionCount = text.count("here") + text.count("there") + text.count("over there") + text.count("beyond") + text.count("nearly") + text.count("opposite") + text.count("under") + text.count("to the left") + text.count("to the right") + text.count("in the distance") 
        #Exemplify count
	#private String[] exemplify = {"chiefly", "especially", "for instance", "in particular", "markedly", "namely", "particularly", "including", "specifically", "such as"};
        Exemplify = 0
        Exemplify = text.count("chiefly") + text.count("especially") + text.count("for instance") + text.count("in particular") + text.count("markedly") + text.count("namely") + text.count("particularly")+ text.count("incluiding") + text.count("specifically") + text.count("such as") 
        #Analytical thinking
        #Analytic = 0 #LIWC Analysis
        #Aunthenticity
        #Authentic  = 0 #LIWC Analysis
        #Emotional tone
        #Tone = 0 #LIWC Analysis
        try:
            #words per sentence (average)
            WPS = 0
            parts = [len(l.split()) for l in re.split(r'[?!.]', text) if l.strip()]
            WPS = sum(parts)/len(parts) #number of words per sentence
        except:
            WPS = 0
        #Six letter words
        Sixltr = 0
        words = text.split()
        letter_count_per_word = {w:len(w) for w in words}
        for x in letter_count_per_word.values():
            if x >= 6:
                Sixltr = Sixltr + 1
        #Function words
        function = 0
        #Pronouns
        pronoun = 0
        text_tokens = word_tokenize(text)
        result = nltk.pos_tag(text_tokens)
        pronoun = len([ (x,y) for x, y in result if y  == "PRP" or y  == "PRP$"])
        #Personal pronouns
        ppron = 0
        ppron = len([ (x,y) for x, y in result if y  == "PRP" ])
        #I
        i = 0
        i = text.count("i")
        #You
        you = 0
        you = text.count("you")
        #Impersonal pronoun "one" / "it"
        ipron = 0
        ipron = text.count("one") + text.count("it")
        #Prepositions
        prep = 0
        prep = len([ (x,y) for x, y in result if y  == "IN" ])
        #Auxiliary verbs do/be/have
        auxverb = 0
        auxverb = text.count("do") + text.count("does") + text.count("don´t") + text.count("doesn´t") + text.count("has") + text.count("have") + text.count("hasn´t")+ text.count("haven´t") + text.count("am") + text.count("are") +  text.count("is") + text.count("´m") + text.count("´re") +  text.count("´s")
        #Negations
        negate = 0
        negate = text.count("not")
        #Count interrogatives
        #interrog = 0 #LICW Analysis
        #Count numbers
        number = 0
        prep = len([ (x,y) for x, y in result if y  == "CD" ])
        #Cognitive processes
        #cogproc = 0 #LIWC Analysis
        #Cause relationships
        #cause = 0 #LIWC Analysis
        #Discrepencies
        #discrep = 0 #LIWC Analysis
        #Tenant
        #tentat = 0 #LIWC Analysis
        #Differtiation
        #differ = 0 #LIWC Analysis
        #Perceptual processes
        #percept = 0 #LIWC Analysis
        #Verbs past focus VBD VBN
        focuspast = 0
        focuspast = len([ (x,y) for x, y in result if y  == "VBN" or y  == "VBD"])
        #Verbs present focus VB VBP VBZ VBG
        focuspresent = 0
        focuspast = len([ (x,y) for x, y in result if y  == "VB" or y  == "VBP" or y  == "VBZ" or y  == "VBG"])
        #net speak
        #netspeak = 0 #LIWC Analysis
        #Assent
        #assent = 0 #LIWC Analysis
        #Non fluencies
        #nonflu = 0 #LIWC Analysis
        #Count all punctuation
        AllPunc = 0
        punctuation = "!#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
        cd = {c:val for c, val in ct.Counter(text).items() if c in punctuation}
        for x in cd.values():
            AllPunc = AllPunc + x
        #number of commas
        Comma = 0
        Comma = text.count(",")
        #number of question marks
        QMark = 0
        QMark = text.count("?")
        
        #return numpy.array([wordCount,readabilityScore,ReadabilityGrade,DiractionCount,Analytic,Authentic,Tone,WPS,Sixltr,function,pronoun,ppron,i,you,ipron,prep,auxverb,negate,interrog,number,cogproc,cause,discrep,tentat,differ,percept,focuspast,focuspresent,netspeak,assent,nonflu,AllPunc,Comma,QMark,Exemplify])
        return numpy.array([wordCount, readabilityScore, ReadabilityGrade, DiractionCount, WPS, Sixltr, pronoun, ppron, i, you, ipron, prep, auxverb, negate, number, focuspast, focuspresent, AllPunc, Comma, QMark,Exemplify])


#Graphic Interface
#Labels
app.addLabel("Title", "Persuasion tool")
#Text Area
app.addScrolledTextArea("Add text here")
#Labels
app.addLabel("selection", "Please select the model that you would like to use for the prediction")
#Radio buttons
app.addCheckBox("Logistic Regression")
app.setCheckBox("Logistic Regression")
app.addCheckBox("Linear Discriminant Analysis")
app.addCheckBox("KNeighbors Classifier")
app.addCheckBox("Decision Tree Classifier")
app.addCheckBox("Gaussian Naive Bayes")
app.addCheckBox("Support Vector Machine")


app.addButtons(["Submit", "Cancel"], press)
app.go()
