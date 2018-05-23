import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.externals import joblib

url = "dataset.csv"
#names = ["http","WC","Analytic","Authentic","Tone","WPS","Sixltr","function","pronoun","ppron","i","you","ipron","prep","auxverb","negate","interrog","number","cogproc","cause","discrep","tentat","differ","percept","focuspast","focuspresent","netspeak","assent","nonflu","AllPunc","Comma","Colon","QMark","Dash","Parenth","OtherP","TFIDF","WordCount","readabilityScore","ReadabilityGrade","Exemplify","Decision"]
names = ["http","WC","Analytic","Authentic","Tone","WPS","Sixltr","function","pronoun","ppron","i","you","ipron","prep","auxverb","negate","interrog","number","cogproc","cause","discrep","tentat","differ","percept","focuspast","focuspresent","netspeak","assent","nonflu","AllPunc","Comma","Colon","QMark","Dash","Parenth","OtherP","TFIDF","WordCount","readabilityScore","ReadabilityGrade","Exemplify","WP","Decision"]
dataset = pandas.read_csv(url, names=names)
dataset.apply(pandas.to_numeric)

# shape
print(dataset.shape)

#Shuffle the dataset
df = shuffle(dataset)

# head
print(df.head(20))

#set training and testing dataset (70-30)
array = df.values
X = array[:,0:40]
Y = array[:,42]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#X_train = X_train.astype('float')
#Y_train = Y_train.astype('float')
#X_validation = X_validation.astype('float')
#Y_validation = Y_validation.astype('float')
seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=4, random_state=None)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Saving the best model Gausian 
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'KNeighborsClassifier.sav'
joblib.dump(model, filename)

filename = 'KNeighborsClassifier.sav'
loaded_model = joblib.load(filename)
result = loaded_model.score(X_validation, Y_validation)
print(result)

