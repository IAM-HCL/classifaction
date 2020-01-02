import numpy as np
import os
import tensorflow as tf

import tqdm
import matplotlib.pyplot as plt

#len(vocab_to_int) = 21337
embed_size = 300 
word_num = 21337
seq_len = 52
lstm_hiden_size = 512
lstm_hiden_layers = 1
batch_size = 128
learning_rate = 0.001
keep_prob_num = 0.5
EPOCHES = 1

train_fea = np.load("./data/trainFea.npy")
train_label = np.load("./data/trainLabel.npy")
dev_fea = np.load("./data/devFea.npy")
dev_label = np.load("./data/devLabel.npy")
test_fea = np.load("./data/testFea.npy")
test_label = np.load("./data/testLabel.npy")


static_embeddings = np.load("./data/static_embeddings.npy")
print("data finished\ndata finished\ndata finished\n\n")


tf.reset_default_graph()

def get_batch(x, y):
    global batch_size
    n_batches = int(x.shape[0] / batch_size)
    
    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]
    
        yield x_batch, y_batch


with tf.name_scope("rnn"):
    # placeholders
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, seq_len), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
    # embeddings
    with tf.name_scope("embeddings"):
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
    
    # model
    with tf.name_scope("model"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_hiden_size)
        
        drop_lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        #cell = tf.contrib.rnn.MultiRNNCell([drop_lstm]*lstm_hiden_layers)
        cell = tf.nn.rnn_cell.MultiRNNCell([ tf.nn.rnn_cell.BasicLSTMCell(lstm_hiden_size) for _ in range(lstm_hiden_layers)])
        initial_state_l = cell.zero_state(batch_size,tf.float32)
        ot, lstm_state = tf.nn.dynamic_rnn(cell=cell,inputs=embed, initial_state=initial_state_l)
        
        #print(lstm_state[0].h)
        W = tf.Variable(tf.truncated_normal((lstm_hiden_size, 1), mean=0.0, stddev=0.1), name="W")
        b = tf.Variable(tf.zeros(1), name="b")
        
        logits = tf.add(tf.matmul(lstm_state[0].h, W), b)
        outputs = tf.nn.sigmoid(logits, name="outputs")
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    
    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # evaluation
    with tf.name_scope("evaluation"):
        #correct_preds = tf.equal(tf.cast(tf.greater(outputs, 0.5), tf.float32), targets)
        #accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))
        max_pool = tf.reduce_max(ot,reduction_indices=[1])
        predictions = tf.contrib.layers.fully_connected(max_pool, 1, activation_fn=tf.sigmoid)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(predictions), tf.float32), targets), tf.float32))


rnn_train_accuracy = []
rnn_test_accuracy = []



saver = tf.train.Saver()



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter("./graphs/rnn", tf.get_default_graph())
    n_batches = int(train_fea.shape[0] / batch_size)
    m_batches = int(dev_fea.shape[0] / batch_size)
    for epoch in range(EPOCHES):
        total_loss = 0
        iteration = 0
        dev_acc = 0.0
        dev_cnt = 0
        dev_cnt_real = 0
        for x_batch, y_batch in get_batch(train_fea, train_label):
            _, l = sess.run([optimizer, loss], feed_dict={inputs: x_batch, targets: y_batch[:,None], keep_prob:keep_prob_num})#'''np.reshape(y_batch, (-1,1))'''
            total_loss += l
            iteration = iteration + 1 
        for x_dbatch, y_dbatch in get_batch(dev_fea, dev_label):
            batch_acc = sess.run([accuracy], feed_dict={inputs:x_dbatch, targets:y_dbatch[:,None],keep_prob:1.0})
            dev_cnt = dev_cnt + 1
            if(dev_cnt > -1):
                dev_cnt_real = dev_cnt_real + 1
                dev_acc = dev_acc + batch_acc[0]
                print(batch_acc)
        '''
        train_corrects = sess.run(accuracy, feed_dict={inputs: train_fea, targets: np.reshape(train_label,  (-1,1) )})
        train_acc = train_corrects / train_fea.shape[0]
        rnn_train_accuracy.append(train_acc)

        test_corrects = sess.run(accuracy, feed_dict={inputs: dev_fea, targets:np.reshape(dev_label,  (-1,1) ) })
        test_acc = test_corrects / dev_fea.shape[0]
        rnn_test_accuracy.append(test_acc)
        '''

        #print("dev_cnt_real "+str(dev_cnt_real))
        dev_acc = dev_acc/dev_cnt_real
        print("Training epoch: {}, Train loss: {:.4f},  dev accuracy: {:.4f}".format(epoch + 1,  total_loss / n_batches, dev_acc))
    
    saver.save(sess, "checkpoints/rnn")
    writer.close()


# INSss[66]:


plt.plot(rnn_train_accuracy)
plt.plot(rnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of LSTM model")
plt.legend(["train", "test"])

dev_cnt= 0
dev_cnt_real = 0
dev_acc = 0.0
print("test:-----------------------")
with tf.Session() as sess:
    saver.restore(sess, "checkpoints/rnn")
    for x_tbatch, y_tbatch in get_batch(test_fea, test_label):
            batch_acc = sess.run([accuracy], feed_dict={inputs:x_tbatch, targets:y_tbatch[:,None],keep_prob:1.0})
            dev_cnt = dev_cnt + 1
            if(dev_cnt > -1):
                dev_cnt_real = dev_cnt_real + 1
                dev_acc = dev_acc + batch_acc[0]
                print(batch_acc[0])

dev_acc = dev_acc/dev_cnt_real
print("Test accuracy: {:.4f}".format(dev_acc))