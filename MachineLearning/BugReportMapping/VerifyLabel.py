import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
#from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf


try:
    engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/eclipse_platform_ui', echo=False)

    df_ui = pd.read_sql_table('bug_and_files', engine)
except:
    df_ui = pd.read_csv("eclipse_platform_ui_no_space.csv")

rows, cols = df_ui.shape

print(df_ui.shape)
labels = []
labels_verified = set()



for i in range(2):
    label = df_ui['files'][i]
    label = re.sub('\.', '_', label)
    label = re.sub('/', '__', label)
    labels_verified.update(label.split('\n'))
    labels.append(label)

print('size of labels_verified: ')
print(len(labels_verified))

print('size of labels: ')
print(len(labels))

countVectorizer = CountVectorizer()

y_array = countVectorizer.fit_transform(labels).toarray()

print('size of y_array: ')
print(y_array.shape)
#y_verify = countVectorizer.fit_transform(labels_verified).toarray()
#print('size of y_verify: ')
#print(y_verify.shape)
feature_names = countVectorizer.get_feature_names()
print(feature_names)



