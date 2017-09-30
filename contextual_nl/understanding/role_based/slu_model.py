import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class slu_model(object):
    def __init__(self, max_seq_len, intent_dim):
        self.hidden_size = 128
        self.intent_dim = intent_dim # one hot encoding
        self.embedding_dim = 200 # read from glove
        self.total_word = 400002 # total word embedding vectors
        self.max_seq_len = max_seq_len
        self.filter_sizes = [2,3,4]
        self.filter_depth = 128
        self.hist_len = 3
        self.add_variables()
        self.add_placeholders()
        self.add_variables()
        self.build_graph()
        self.add_loss()
        self.add_train_op()
        self.init_embedding()
        self.init_model = tf.global_variables_initializer()

    def init_embedding(self):
        self.init_embedding = self.embedding_matrix.assign(self.read_embedding_matrix)

    def add_variables(self):
        self.embedding_matrix = tf.Variable(tf.truncated_normal([self.total_word, self.embedding_dim]), dtype=tf.float32, name="glove_embedding")

    def add_placeholders(self):
        self.history_intent = tf.placeholder(tf.float32, [None, self.hist_len * 2, self.intent_dim])
        self.tourist_input_intent, self.guide_input_intent = tf.split(self.history_intent, num_or_size_splits=2, axis=1)
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        self.current_nl_len = tf.placeholder(tf.int32, [None])
        self.current_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.hist_nl = tf.placeholder(tf.int32, [None, self.hist_len * 2, self.max_seq_len])
        self.tourist_input_nl, self.guide_input_nl = tf.split(self.hist_nl, num_or_size_splits=2, axis=1)
        self.hist_nl_len = tf.placeholder(tf.int32, [None, self.hist_len * 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def hist_cnn(self, scope, idx):
        with tf.variable_scope(scope):
            if idx != 0:
                tf.get_variable_scope().reuse_variables()
            if scope == 'tourist':
                # tourist_input_nl should now have [batch_size, 3, max_seq_len]
                tourist_input_nl = tf.unstack(self.tourist_input_nl, axis=1)[idx]
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, tourist_input_nl)
            elif scope == 'guide':
                guide_input_nl = tf.unstack(self.guide_input_nl, axis=1)[idx]
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, guide_input_nl)
            pooled_outputs = list()
            for idx, filter_size in enumerate(self.filter_sizes):
                # convolution layer
                kernel_size = (filter_size)
                h = tf.layers.conv1d(inputs, self.filter_depth, kernel_size, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
                # max over time pooling
                pooled = tf.layers.max_pooling1d(h, (self.max_seq_len-filter_size+1), 1)
                pooled_outputs.append(pooled)
            num_filters_total = self.filter_depth * len(self.filter_sizes)
            h_pool_flat = tf.squeeze(tf.concat(pooled_outputs, axis=2), axis=1)
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            dense_h = tf.layers.dense(inputs=h_drop, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
            return dense_h

    def nl_biRNN(self, history_summary):
        with tf.variable_scope("nl"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            history_summary = tf.expand_dims(history_summary, axis=1)
            replicate_summary = tf.tile(history_summary, [1, self.max_seq_len, 1]) # [batch_size, self.max_seq_len, self.intent_dim]
            concat_input = tf.concat([inputs, replicate_summary], axis=2) # [batch_size, self.max_seq_len, self.intent_dim+self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, concat_input, sequence_length=self.current_nl_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            return outputs
    
    def role_model(self):
        self.unstack_tourist_cnn_hist = list()
        self.unstack_guide_cnn_hist = list()
        for i in range(self.hist_len):
            self.unstack_tourist_cnn_hist.append(self.hist_cnn('tourist', i))
            self.unstack_guide_cnn_hist.append(self.hist_cnn('guide', i))
        
        stack_tourist_cnn_hist = tf.sigmoid(tf.stack(self.unstack_tourist_cnn_hist, axis=1))
        stack_guide_cnn_hist = tf.sigmoid(tf.stack(self.unstack_guide_cnn_hist, axis=1))
        
        # BLSTM for 2 roles
        with tf.variable_scope("tourist_nl"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, stack_tourist_cnn_hist, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            tourist_outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states

        with tf.variable_scope("guide_nl"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, stack_guide_cnn_hist, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            guide_outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states

        return tf.concat([tourist_outputs, guide_outputs], axis=1)

    def build_graph(self):
        concat_output = self.role_model()
        history_summary = tf.layers.dense(inputs=concat_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        final_output = self.nl_biRNN(history_summary)
        self.output = tf.layers.dense(inputs=final_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        self.intent_output = tf.sigmoid(self.output)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.output))

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)
