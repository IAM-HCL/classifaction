import numpy as np
import os
import tensorflow as tf

#len(vocab_to_int) = 21337
word_num = 21337
seq_len = 52
lstm_hiden_size = 256
lstm_hiden_layers = 1
batch_size = 128
learning_rate = 0.001

train_fea = np.load("./data/trainFea.npy")
train_label = np.load("./data/trainLabel.npy")
dev_fea = np.load("./data/devFea.npy")
dev_label = np.load("./data/devLabel.npy")

tf.reset_default_graph()
X = tf.placeholder(tf.int32,[None,seq_len],name='inputs')
labels_ = tf.placeholder(tf.int32,[None,1],name='labels')
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

embed_size = 300 

embedding = tf.Variable(tf.random_uniform((word_num,embed_size),-1,1))
embed = tf.nn.embedding_lookup(embedding,X)


lstm = tf.contrib.rnn.BasicLSTMCell(lstm_hiden_size)

drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)

cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_hiden_layers)

initial_state = cell.zero_state(batch_size,tf.float32)

outputs,final_state = tf.nn.dynamic_rnn(cell=cell,inputs=embed,initial_state=initial_state)

max_pool = tf.reduce_max(outputs,reduction_indices=[1])
predictions = tf.contrib.layers.fully_connected(max_pool, 1, activation_fn=tf.sigmoid)
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels_, predictions)
tf.summary.scalar('cost',cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_), tf.float32))
tf.summary.scalar('accuracy',accuracy)
def get_batches(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

merged = tf.summary.merge_all()
direc = './graph/'
train_writer = tf.summary.FileWriter(direc+'train',tf.get_default_graph())
test_writer = tf.summary.FileWriter(direc+'test',tf.get_default_graph())

epochs = 2
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        
        for ii, (x, y) in enumerate(get_batches(train_fea, train_label, batch_size), 1):
            feed = {X: x,
                    labels_: y[:,None],
                    keep_prob:0.6}
            loss, _, summary1 = sess.run([cost, optimizer, merged], feed_dict=feed)
            
            if iteration%5==0:
                train_writer.add_summary(summary1,iteration)
                print("Epoch: {}/{}".format(e+1, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%10==0:
                val_acc = []
                for x, y in get_batches(dev_fea, dev_label, batch_size):
                    feed = {X: x,
                            labels_: y[:,None],
                            keep_prob:1.0}
                    batch_acc, summary2 = sess.run([accuracy, merged], feed_dict=feed)
                    val_acc.append(batch_acc)
                test_writer.add_summary(summary2,iteration)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "./model/sentiment.ckpt")