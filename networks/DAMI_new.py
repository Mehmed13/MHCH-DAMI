#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.initializers import GlorotNormal
from utility import *
from networks.layers.attention import *
from networks.layers.transformer import *

class DAMI(tf.keras.Model):
    def __init__(self, memory=0, vocab=None, config_dict=None, **kwargs):
        super(DAMI, self).__init__()
        self.model_name = self.__class__.__name__
        self.logger = logging.getLogger("Tensorflow")
        self.logger.info(f"Model Name: {self.model_name}")

        # Configuration parameters
        self.rnn_dim = config_dict["rnn_dim"]
        self.dense_dim = config_dict["dense_dim"]
        self.pos_dim = config_dict["pos2id"]
        self.lr = config_dict["learning_rate"]
        self.l2_reg_lambda = config_dict["l2_reg_lambda"]
        self.weight_decay = config_dict["weight_decay"]
        
        # Data parameters
        self.dia_max_len = 30
        self.sent_max_len = 50
        self.nb_classes = 2
        self.embedding_dim = 200
        
        # Initialize layers
        self.embedding_layer = layers.Embedding(
            input_dim=vocab.size() + 1,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.constant_initializer(vocab.embeddings) if vocab.embeddings is not None else GlorotNormal(),
            trainable=True
        )
        
        self.dropout = Dropout(0.5)
        
        # Attention layers
        self.attention_agent = layers.Dense(2 * self.rnn_dim, activation='tanh')
        self.attention_customer = layers.Dense(2 * self.rnn_dim, activation='tanh')
        
        # RNN layers
        self.bidirectional_rnn = Bidirectional(LSTM(self.rnn_dim, return_sequences=True, return_state=True))
        self.dialogue_rnn = LSTM(self.rnn_dim, return_sequences=True)
        
        # Dense layers
        self.local_inference = Dense(self.dense_dim, activation='relu')
        self.context_aware = Dense(self.dense_dim, activation='relu')
        self.output_layer = Dense(self.nb_classes, activation='softmax')
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=self.lr)

    def call(self, inputs, training=False):
        # Unpack inputs
        input_x1, input_x2, input_x3, tfs, pos_list, sent_len, dia_len = inputs
        
        # Embedding layer
        embedded = self.embedding_layer(input_x1)
        if training:
            embedded = self.dropout(embedded)
            tfs = self.dropout(tfs)
        
        # Reshape inputs
        embedded_reshaped = tf.reshape(embedded, [-1, self.sent_max_len, self.embedding_dim])
        tfs_reshaped = tf.expand_dims(tf.reshape(tfs, [-1, self.sent_max_len]), axis=-1)
        pos_list_reshaped = tf.reshape(pos_list, [-1, self.sent_max_len, self.pos_dim])
        sent_len_reshape = tf.reshape(sent_len, [-1])
        
        # Positional encoding
        sent_pos_emb = positional_encoding(embedded_reshaped, self.sent_max_len)
        embedded_all = tf.concat([embedded_reshaped, sent_pos_emb, pos_list_reshaped], axis=-1)
        
        # Sentence encoding
        sent_encoder_output, forward_state, backward_state = self.bidirectional_rnn(embedded_all)
        sent_encoder_state = tf.concat([forward_state[-1], backward_state[-1]], axis=-1)
        
        # Attention mechanism
        agent_tag = tf.reshape(input_x2, [-1])
        customer_tag = 1 - agent_tag
        
        # Agent attention
        qk_a = self.attention_agent(embedded_all)
        vu_a = tf.nn.relu(tf.reduce_sum(qk_a * (1 - tfs_reshaped), axis=-1))
        dif_mask = tf.sequence_mask(sent_len_reshape, maxlen=self.sent_max_len, dtype=tf.float32)
        sent_paddings = tf.ones_like(dif_mask) * (-2**32+1)
        vu_masked_a = tf.where(tf.equal(dif_mask, 0), sent_paddings, vu_a)
        vu_masked_a = tf.where(tf.equal(agent_tag, 0), sent_paddings, vu_masked_a)
        alphas_a = tf.nn.softmax(vu_masked_a)
        
        # Customer attention
        qk_c = self.attention_customer(embedded_all)
        vu_c = tf.reduce_sum(qk_c * (1 - tfs_reshaped), axis=-1)
        vu_masked_c = tf.where(tf.equal(dif_mask, 0), sent_paddings, vu_c)
        vu_masked_c = tf.where(tf.equal(customer_tag, 0), sent_paddings, vu_masked_c)
        alphas_c = tf.nn.softmax(vu_masked_c)
        
        # Combine attention
        alphas = alphas_a + alphas_c
        sent_encoder_output_atten = tf.reduce_sum(sent_encoder_output * tf.expand_dims(alphas, -1), 1)
        
        # Reshape for dialogue processing
        sent_encoder_attened_reshape = tf.reshape(sent_encoder_output_atten, [-1, self.dia_max_len, 2 * self.rnn_dim])
        sent_encoder_state_reshape = tf.reshape(sent_encoder_state, [-1, self.dia_max_len, 2 * self.rnn_dim])
        
        # Combine embeddings
        senti_score = tf.expand_dims(input_x3, axis=-1)
        combine_emb = tf.concat([sent_encoder_attened_reshape, sent_encoder_state_reshape, senti_score], axis=-1)
        if training:
            combine_emb = self.dropout(combine_emb)
        
        # Local inference
        cross_match = tf.matmul(combine_emb, tf.transpose(combine_emb, [0, 2, 1]))
        cross_match_upper = tf.linalg.band_part(cross_match, 0, -1)
        cross_match_sim_one_direct = cross_match - cross_match_upper
        encode_local = tf.concat([combine_emb, cross_match_sim_one_direct], axis=-1)
        encode_local = self.local_inference(encode_local)
        if training:
            encode_local = self.dropout(encode_local)
        
        # Dialogue encoding
        v_dia_encode_output = self.dialogue_rnn(encode_local)
        
        # Context-aware attention
        dia_seq_mask = tf.sequence_mask(dia_len, maxlen=self.dia_max_len, dtype=tf.float32)
        w_datt = tf.Variable(tf.random.normal([self.rnn_dim, self.rnn_dim]))
        d_att = tf.matmul(tf.matmul(v_dia_encode_output, w_datt), tf.transpose(v_dia_encode_output, [0, 2, 1]))
        d_att_mask = tf.tile(tf.expand_dims(dia_seq_mask, axis=-1), [1, 1, self.dia_max_len])
        paddings = tf.ones_like(d_att) * (-2**32+1)
        a_att_masked = tf.where(tf.equal(d_att_mask, 0), paddings, d_att)
        d_att_alpha = tf.nn.softmax(tf.linalg.band_part(a_att_masked, -1, 0))
        
        context_encode = tf.matmul(d_att_alpha, v_dia_encode_output)
        concat_d = tf.concat([context_encode, v_dia_encode_output], axis=-1)
        combine_h = self.context_aware(concat_d)
        if training:
            combine_h = self.dropout(combine_h)
        
        # Output layer
        logits = self.output_layer(combine_h)
        output = tf.argmax(logits, axis=-1)
        proba = tf.nn.softmax(logits)
        
        return logits, output, proba

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, _, _ = self(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits))
            if self.l2_reg_lambda > 0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
                loss += self.l2_reg_lambda * l2_loss
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss}

    def test_step(self, data):
        x, y = data
        logits, _, _ = self(x, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits))
        return {'loss': loss} 