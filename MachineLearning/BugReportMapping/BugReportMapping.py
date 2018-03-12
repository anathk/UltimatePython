import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

try:
    engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/eclipse_platform_ui', echo=False)

    df_ui = pd.read_sql_table('bug_and_files', engine)
except:
    df_ui = pd.read_csv("eclipse_platform_ui_no_space.csv")

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
    report = re.sub('[^a-zA-Z0-9]', ' ', (df_ui['summary'][i]))
    label = df_ui['files'][i]
    corpus.append(sentence_to_stemmed(report))
    label = re.sub('\.', '_', label)
    label = re.sub('/', '__', label)
    labels.append(label)
    if i % 100 == 0:
        print('%d lines processed.\n' % i)

countVectorizer = CountVectorizer()

X_array = countVectorizer.fit_transform(corpus).toarray()
# y_array = countVectorizer.fit_transform(labels).toarray()

max_document_length_y = max([len(x.split("\n")) for x in labels])
vocab_processor_y = learn.preprocessing.VocabularyProcessor(max_document_length_y)
y_array = np.array(list(vocab_processor_y.fit_transform(labels)))

X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(X_array, y_array, test_size = 0.2, random_state =0)

# pca = PCA(n_components=None)
# X_train_array = pca.fit_transform(X_train_array)
# X_test_array = pca.transform(X_test_array)
# explained_variance = pca.explained_variance_ratio_
print(X_train_array.shape)


X = tf.placeholder(tf.float32, [5196, 11045])
W = tf.Variable(tf.zeros([11045, 383]))
b = tf.Variable(tf.zeros([5196]))

#tf.add(tf.matmul(images, weights1), bias1
y = tf.add(tf.matmul(X, W), b)



y_ = tf.placeholder(tf.float32, [5196, 383])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.global_variables_initializer().run()
  # Train
for _ in range(2):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: X_train_array, y_: y_train_array})


correct_prediction = tf.nn.in_top_k(tf.cast(tf.argmax(y, 1), tf.float32), y_, 5)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={X: X_test_array,
                                      y_: y_test_array}))

