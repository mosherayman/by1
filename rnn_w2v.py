'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import word2vec

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = 'belling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    content = np.array([i for i in content if i not in [',','.','?']])
    return content

training_data = read_data(training_file)
print("Loaded training data...")

vector_size=100
def build_vectors():
    #word2vec.word2vec(fname, './text8.bin', size=vector_size, verbose=True)
    model = word2vec.load('/data/text8.bin')
    return model

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

model = build_vectors()
#vocab_size = len(model.vocab)
#using vector_size instead of vocab_size

# Parameters
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 3
#vector size is set above

# number of units in RNN cell
n_hidden = 512
#n_hidden = 1024

# tf Graph input
#x = tf.placeholder("float", [None, n_input, 1])
x = tf.placeholder("float", [None, n_input, vector_size])
#y = tf.placeholder("float", [None, vector_size])
y = tf.placeholder("float", [None, vocab_size]) #for onehot

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):

    #x is [None/batch_size, n_input, vector_size]
    # Unstack to get a list of 'n_input' tensors of shape (batch_size, vector_size)
    x = tf.unstack(x, n_input, 1)

    # reshape to [1, n_input]
    #x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

#Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#correct_pred = tf.square(pred - y)
#correct_pred = tf.equal(pred, y)
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()  

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [model[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vector_size])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        #symbols_out_onehot[model.vocab_hash[str(training_data[offset+n_input_words])]] = 1.0
        #symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        #symbols_out = model[training_data[offset+n_input_words]]
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            #symbols_out_pred = model.vocab[int(tf.argmax(onehot_pred, 1).eval())]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            #metrics = np.dot(model.vectors, onehot_pred.T)
            #best = np.argsort(metrics)[::-1][1:2][0]
            #symbols_out_pred = model.vocab[best]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    print("saving model")
    saver.save(session, 'my-model')  
    print("done saving model")
    while False:
        print("testing model from memory")
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        #try:
        if True:
            symbols_in_keys = [model[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vector_size])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                #metrics = np.dot(model.vectors, vec_pred.T)
                #best = np.argsort(metrics)[::-1][1:10+1]
                #vocab_index = best[0]
                pred_word = reverse_dictionary[onehot_pred_index]
                sentence = "%s %s" % (sentence, pred_word)
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(model[pred_word])
            print(sentence)
        #except:
            #print("Word not in dictionary")

    offset = 0
    sentence=[]
    while offset < len(training_data)-n_input:
        next_3_words= [ str(training_data[i]) for i in range(offset, offset+n_input) ]
        symbols_in_keys = [ model[ str(training_data[i])] for i in range(offset, offset+n_input) ]
        keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, vector_size])
        onehot_pred = session.run(pred, feed_dict={x: keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        if onehot_pred_index == dictionary[training_data[offset+n_input]]:
              sentence.append(training_data[offset+n_input])
        else:
              sentence.append("(%s) [%s]" % (reverse_dictionary[onehot_pred_index], training_data[offset+n_input].upper()))
        print("%s %s %s %s" % ((onehot_pred_index == dictionary[training_data[offset+n_input]]),",".join(next_3_words), training_data[offset+n_input], reverse_dictionary[onehot_pred_index]))
        offset += 1
        #offset += n_input+1
    print (" ".join(sentence))
