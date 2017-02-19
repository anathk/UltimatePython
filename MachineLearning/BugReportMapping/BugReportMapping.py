import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf


engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/eclipse_platform_ui', echo=False)

dataset = pd.read_sql_table('bug_and_files', engine)

corpus = []
labels = []

def sentence_to_stemmed(sentence):
    sentence = sentence.lower()
    sentence = sentence.split()
    porterStemmer = PorterStemmer()
    sentence = [porterStemmer.stem(word) for word in sentence if not word in set(stopwords.words('english'))]
    sentence = ' '.join(sentence)

    return sentence

for i in range(6495):
    report = re.sub('[^a-zA-Z0-9]', ' ', (dataset['summary'][i]))
    label = dataset['files'][i]
    corpus.append(sentence_to_stemmed(report))
    label = re.sub('\.', '_', label)
    label = re.sub('/', '__', label)
    labels.append(label)
    if i % 100 == 0:
        print('%d lines processed.\n' % i)

countVectorizer = CountVectorizer()

X_array = countVectorizer.fit_transform(corpus).toarray()
y_array = countVectorizer.fit_transform(labels).toarray()

X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(X_array, y_array, test_size = 0.2, random_state =0)


X = tf.placeholder(tf.float32, [None, 11045])
W = tf.Variable(tf.zeros([11045, 6122]))
b = tf.Variable(tf.zeros([6122]))

y = tf.matmul(X, W) + b

y_ = tf.placeholder(tf.float32, [None, 6122])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
  # Train
for _ in range(2):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: X_train_array, y_: y_train_array})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={X: X_test_array,
                                      y_: y_test_array}))

