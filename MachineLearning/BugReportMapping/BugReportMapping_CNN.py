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
from TextCNN import TextCNN

import tensorflow as tf
import numpy as np
import os
import time
import datetime





def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)

    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def sentence_to_stemmed(sentence):
    sentence = sentence.lower()
    sentence = sentence.split()
    porterStemmer = PorterStemmer()
    sentence = [porterStemmer.stem(word) for word in sentence if not word in set(stopwords.words('english'))]
    sentence = ' '.join(sentence)

    return sentence




try:
    engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/eclipse_platform_ui', echo=False)

    df_ui = pd.read_sql_table('bug_and_files', engine)
except:
    df_ui = pd.read_csv('eclipse_platform_ui_no_space.csv')

corpus = []
labels = []

rows, cols = df_ui.shape

for i in range(rows):
    report = re.sub('[^a-zA-Z0-9]', ' ', (df_ui['summary'][i]))
    label = df_ui['files'][i]


    label = re.sub('\.', '_', label)
    label = re.sub('/', '__', label)
    if len(label.split('\n')) == 1:
        corpus.append(sentence_to_stemmed(report))
        labels.append(label)


    if i % 100 == 0:
        print('%d lines processed.\n' % i)

print('Total %d lines in corpus.' % len(labels))


max_document_length_x = max([len(x.split(" ")) for x in corpus])
max_document_length_y = max([len(x.split("\n")) for x in labels])
countVectorizer = CountVectorizer()
vocab_processor_x = learn.preprocessing.VocabularyProcessor(max_document_length_x)
#x= countVectorizer.fit_transform(corpus).toarray()
x = np.array(list(vocab_processor_x.fit_transform(corpus)))

#y = countVectorizer.fit_transform(labels).toarray()

vocab_processor_y = learn.preprocessing.VocabularyProcessor(max_document_length_y)
y = np.array(list(vocab_processor_y.fit_transform(labels)))

print('x shape: %d, %d' % x.shape)
print('y shape: %d, %d' % y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

num_filters = 128
l2_reg_lambda = 0.03
dropout_keep_prob = 0.5
batch_size = 4
num_epochs = 100
evaluate_every = 100
embedding_dim = 128
filter_sizes = "5"

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    #sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(x),
            embedding_size=embedding_dim,
            filter_sizes=list(map(int, filter_sizes.split(","))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        ## Define Training procedure
        train_op = tf.train.AdamOptimizer(1e-3).minimize(cnn.loss)

        ## Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        ## Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        ## Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        ## Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch, step):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, summaries, loss, accuracy = sess.run(
                [train_op, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, step, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            summaries, loss, accuracy = sess.run(
                [dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
        # Training loop. For each batch...
        current_step = 0
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, current_step)
            current_step += 1
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_test, y_test, current_step, writer=dev_summary_writer)
                print("")