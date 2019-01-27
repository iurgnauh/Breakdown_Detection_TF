import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


class TextRNN:
    def __init__(self, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, vocab_size,
                 embed_size, hidden_size, dropout_keep_prob, is_training, tran_layer=13, attention=False, input_embedding=None, glove_init=None, initializer=tf.random_normal_initializer(stddev=0.1)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.dropout_keep_prob = dropout_keep_prob
        self.initializer = initializer
        self.num_sampled = 20

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")  # y [None,num_classes]
        self.mask_label = tf.placeholder(tf.float32, [None], name="mask_label")
        self.seq_len_list = tf.placeholder(tf.int32, [None], name="seq_len_list")
        # ------------------------
        self.glove_embedding = tf.get_variable("glove_embedding", [vocab_size, embed_size], initializer=glove_init)
        self.glove_init = True if glove_init else False
        self.input_embedding = input_embedding
        self.attention = attention
        # ------------------------

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.Embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embed_size]), trainable=False, name="Embedding")
        self.Embed_placeholder = tf.placeholder(tf.float32, [vocab_size, embed_size], name="Embed_placeholder")
        self.embedding_init = self.Embedding.assign(self.Embed_placeholder)

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        self.loss_val = self.loss() #-->self.loss_nce()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        # correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
        self.merged = tf.summary.merge_all()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])       #[label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """
        # 1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x) #shape:[None,sentence_length,embed_size]
        if self.glove_init:
            self.embedded_words = tf.nn.embedding_lookup(self.glove_embedding, self.input_x)
        # ------------------------
        if self.attention:
            glove_emb = tf.nn.embedding_lookup(self.glove_embedding, self.input_x)  # [batch, sentence_length, embed_size]
            # input_embedding: [batch, sentence_length, trans_layer, embed_size]
            weights = tf.nn.softmax(tf.squeeze(tf.matmul(self.input_embedding, tf.expand_dims(glove_emb, -1)), -1), -1)  # [batch, sentence_length, tran_layer]
            self.embedded_words = tf.reduce_sum(self.input_embedding * tf.expand_dims(weights, -1), 2)  # [batch, sentence_length, embed_size]

        # ------------------------
        # 2. Bi-lstm layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, sequence_length=self.seq_len_list, dtype=tf.float32)  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        # print("outputs:===>",outputs) #outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))

        # 3. concat output
        output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        # Select last hidden state before the padding, https://danijar.com/variable-sequence-lengths-in-tensorflow/
        batch_size = tf.shape(output_rnn)[0]
        max_length = tf.shape(output_rnn)[1]
        out_size = int(output_rnn.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (self.seq_len_list - 1)
        flat = tf.reshape(output_rnn, [-1, out_size])
        self.output_rnn_last = tf.gather(flat, index)  # [batch_size,hidden_size*2]

        # 4. logits(use linear layer)
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)  # [batch_size]
            loss = tf.reduce_mean(losses * self.mask_label)  # print("2.loss.loss:", loss) #shape=()
            # Doing Regularization
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        tf.summary.scalar('loss', loss)
        return loss

    def loss_nce(self,l2_lambda=0.0001):  # 0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
        if self.is_training: #training
            # labels=tf.reshape(self.input_y,[-1])               #[batch_size,1]------>[batch_size,]
            labels=tf.expand_dims(self.input_y,1)                   #[batch_size,]----->[batch_size,1]
            loss = tf.reduce_mean( #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),#[hidden_size*2, num_classes]--->[num_classes,hidden_size*2]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                               biases=self.b_projection,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                               labels=labels,                 #[batch_size,1]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
                               inputs=self.output_rnn_last,# [batch_size,hidden_size*2] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                               num_sampled=self.num_sampled,  #scalar. 100
                               num_classes=self.num_classes,partition_strategy="div"))  #scalar. 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op


def test():
    # Below is a function test; if you use this for text classifiction, you need to
    # tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes = 3
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1  # 0.5
    textRNN = TextRNN(num_classes, learning_rate, decay_steps, decay_rate, sequence_length, vocab_size,
                      embed_size, dropout_keep_prob, is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.random.rand(8, sequence_length)  # [None, self.sequence_length]
            input_y = np.array(
                [1, 0, 1, 1, 1, 2, 1, 1])  # np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
            loss, predict, _ = sess.run(
                [textRNN.loss_val, textRNN.predictions, textRNN.train_op],
                feed_dict={textRNN.input_x: input_x, textRNN.input_y: input_y})
            print("loss:", loss, "label:", input_y, "prediction:", predict)


if __name__ == '__main__':
    test()
