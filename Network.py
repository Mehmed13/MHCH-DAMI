#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.initializers import GlorotNormal
from sklearn.metrics import auc, roc_curve, confusion_matrix, classification_report
from utility import *

warnings.filterwarnings('ignore', category=FutureWarning)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class Network(tf.keras.Model):
    """
    Parent class for all tensorflow networks.
    """

    def __init__(self, vocab=None, sent_max_len=50, dia_max_len=30, nb_classes=2, nb_words=5000,
                 embedding_dim=200, dense_dim=128, rnn_dim=128, keep_prob=0.5, lr=0.001,
                 weight_decay=0.0, l2_reg_lambda=0.0, optim='adam', gpu='0', memory=0, **kwargs):
        super(Network, self).__init__()
        
        # logging
        self.logger = logging.getLogger("Tensorflow")

        # data config
        self.sent_max_len = sent_max_len
        self.dia_max_len = dia_max_len
        self.nb_classes = nb_classes
        self.nb_words = nb_words

        # network config
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.rnn_dim = rnn_dim
        self.keep_prob = keep_prob

        # optimizer config
        self.lr = lr
        self.weight_decay = weight_decay
        self.l2_reg_lambda = l2_reg_lambda
        self.optim = optim

        self.lamb = 0.

        self.model_name = 'Network'
        self.data_name = 'makeup'

        # session info config
        self.gpu = gpu
        self.memory = memory
        self.vocab = vocab
        print(self.vocab.embeddings.shape)

        # Initialize layers
        self.embedding_layer = layers.Embedding(
            input_dim=self.nb_words,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.constant_initializer(self.vocab.embeddings) if self.vocab.embeddings is not None else GlorotNormal(),
            trainable=True
        )
        
        self.dropout = Dropout(1 - self.keep_prob)
        
        # Initialize optimizer
        if self.optim == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=self.lr)
        else:
            self.optimizer = optimizers.SGD(learning_rate=self.lr)

        # Set up save directory
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    'weights', self.data_name, self.model_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def set_nb_words(self, nb_words):
        self.nb_words = nb_words
        self.logger.info("set nb_words.")

    def set_data_name(self, data_name):
        self.data_name = data_name
        self.logger.info("set data_name.")

    def set_name(self, model_name):
        self.model_name = model_name
        self.logger.info("set model_name.")

    def set_from_model_config(self, model_config):
        self.embedding_dim = model_config['embedding_dim']
        self.optim = model_config['optimizer']
        self.lr = model_config['learning_rate']
        self.weight_decay = model_config['weight_decay']
        self.l2_reg_lambda = model_config['l2_reg_lambda']
        self.logger.info("set from model_config.")

    def set_from_data_config(self, data_config):
        self.nb_classes = data_config['nb_classes']
        self.logger.info("set from data_config.")

    def build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed(self.vocab.embeddings)
        self._inference()
        self._compute_loss()
        self._create_train_op()

        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        print_trainable_variables(output_detail=True, logger=self.logger)
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))
        embedding_param_num = sum([np.prod(self.sess.run(tf.shape(v)))
                                   for v in self.all_params if 'word_embedding' in v.name or 'weights' in v.name])
        self.logger.info('There are {} parameters in the model for word embedding'.format(embedding_param_num))
        pure_param_num = sum([np.prod(self.sess.run(tf.shape(v)))
                              for v in self.all_params if 'word_embedding' not in v.name and 'weights' not in v.name])
        self.logger.info('There are {} parameters in the model without word embedding'.format(pure_param_num))

        # save info
        self.save_dir = self.save_dir + '/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.saver = tf.keras.saving.get_checkpoint_manager(self)

        # initialize the model
        self.optimizer.build(self.trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradients, self.trainable_variables))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.input_x1 = tf.keras.Input(shape=(self.dia_max_len, self.sent_max_len), name='input_x1')
        self.input_x2 = tf.keras.Input(shape=(self.dia_max_len,), name="input_x2")
        self.input_x3 = tf.keras.Input(shape=(self.dia_max_len,), name="input_x3")
        self.sent_len = tf.keras.Input(shape=(self.dia_max_len,), name='input_x_sent_len')
        self.dia_len = tf.keras.Input(shape=(1,), name='input_x_len')
        self.input_y = tf.keras.Input(shape=(self.dia_max_len,), name="input_y")
        self.dropout_keep_prob = tf.keras.Input(shape=(), name="dropout_keep_prob")
        self.logger.info("setup placeholders.")

    def _embed(self, embedding_matrix=np.array([None])):
        """
        The embedding layer
        """
        self.word_embeddings = self.embedding_layer(embedding_matrix)

    def _get_a_p_r_f_sara(self, input_y, prediction, category):
        target_class = np.array(np.argmax(input_y, axis=-1))
        pre_class = np.array(np.argmax(prediction, axis=-1))
        accuracy, precision, recall, f1score, sara_f1score, f0_5score, f2score = get_a_p_r_f_sara(target_class,
                                                                                                  pre_class, category)
        return accuracy, precision, recall, f1score, sara_f1score, f0_5score, f2score

    def _inference(self):
        """
        encode sentence information
        """
        # [B, D_len, S_len, E_dim]
        self.embedded = self.word_embeddings
        if self.dropout_keep_prob != 1:
            self.embedded = self.dropout(self.embedded)

        # [B * D_len, S_len, E_dim]
        self.embedded_reshaped = tf.reshape(self.embedded, shape=[-1, self.sent_max_len, self.embedding_dim])
        self.sent_len_reshape = tf.reshape(self.sent_len, shape=[-1])

        with tf.name_scope('sent_encoding'):
            self.sent_encoder_output, self.sent_encoder_state = self._bidirectional_rnn(self.embedded_reshaped,
                                                                                        self.sent_len_reshape)
            if self.dropout_keep_prob != 1:
                # [B * D_len, 2 * H]
                self.sent_encoder_state = self.dropout(self.sent_encoder_state)

        with tf.name_scope('sequence_context_encoding'):
            # [B, D_len, 2 * H]
            self.sent_encoder_reshape = tf.reshape(self.sent_encoder_state, [-1, self.dia_max_len, 2 * self.rnn_dim])
            # RNN
            cells = [tf.keras.layers.Dropout(self.dropout_keep_prob)(tf.keras.layers.LSTM(self.rnn_dim, state_is_tuple=True)(self.sent_encoder_reshape))]
            rnn_cell = tf.keras.layers.RNN(cells, stateful=True)
            # [B, D_len, H] [B, H]
            self.dia_encoder_output, self.dia_encoder_state = rnn_cell(self.sent_encoder_reshape)

        with tf.name_scope('position_predict'):
            # [B, D_len]
            self.position_dense_prob = tf.keras.layers.Dense(units=self.dia_max_len, activation=tf.nn.softmax)(
                self.dia_encoder_output[:, -1, :])

        with tf.name_scope('dialogue_mask'):

            self.dia_seq_mask = tf.sequence_mask(self.dia_len, maxlen=self.dia_max_len, dtype=tf.float32)
            self.dia_position_pre_masked = self.dia_seq_mask * self.position_dense_prob

        with tf.name_scope("output"):
            self.logits = self.dia_position_pre_masked
            self.output = tf.argmax(self.logits, axis=-1)
            self.proba = tf.nn.softmax(self.logits)

        return self.logits

    def _bidirectional_rnn(self, inputs, length, rnn_type='lstm'):
        if rnn_type == 'lstm':
            fw_rnn_cell = tf.keras.layers.LSTMCell(self.rnn_dim)
            fw_rnn_cell = tf.keras.layers.Dropout(self.dropout_keep_prob)(fw_rnn_cell)
            bw_rnn_cell = tf.keras.layers.LSTMCell(self.rnn_dim)
            bw_rnn_cell = tf.keras.layers.Dropout(self.dropout_keep_prob)(bw_rnn_cell)
            outputs, output_states = \
                tf.keras.layers.Bidirectional(tf.keras.layers.RNN(fw_rnn_cell, stateful=True), backward_layer=tf.keras.layers.RNN(bw_rnn_cell, stateful=True))(inputs)
            fw_state, bw_state = output_states
            fw_c, fw_h = fw_state
            bw_c, bw_h = bw_state
            fw_state, bw_state = fw_h, bw_h

            final_output = tf.concat(outputs, -1)
            final_state = tf.concat([fw_state, bw_state], -1)

        elif rnn_type == 'gru':
            fw_rnn_cell = tf.keras.layers.GRUCell(self.rnn_dim)
            fw_rnn_cell = tf.keras.layers.Dropout(self.dropout_keep_prob)(fw_rnn_cell)
            bw_rnn_cell = tf.keras.layers.GRUCell(self.rnn_dim)
            bw_rnn_cell = tf.keras.layers.Dropout(self.dropout_keep_prob)(bw_rnn_cell)
            outputs, output_states = \
                tf.keras.layers.Bidirectional(tf.keras.layers.RNN(fw_rnn_cell, stateful=True), backward_layer=tf.keras.layers.RNN(bw_rnn_cell, stateful=True))(inputs)

            final_output = tf.concat(outputs, -1)
            final_state = tf.concat(output_states, -1)

        return final_output, final_state

    def _compute_loss(self):
        """
        The loss function
        """

        self.loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.input_y, self.logits))
        self.logger.info("Calculate Loss.")

        if self.l2_reg_lambda > 0:
            l2_loss = tf.add_n([tf.keras.regularizers.l2(self.l2_reg_lambda)(v) for v in self.trainable_variables])
            self.loss += l2_loss
            self.logger.info("Add L2 Loss.")

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """

        self.train_op = self.optimizer(self.loss)

    def train(self, data_generator, keep_prob, epochs, data_name,
              mode='train', batch_size=20, nb_classes=2, shuffle=True,
              is_val=True, is_test=True, save_best=True,
              val_mode='eval', test_mode='test'):

        print('Using train DAMI trainer without CRF.')
        max_val = 0

        for epoch in range(epochs):
            self.logger.info('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))
            print('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))
            counter, total_loss = 0, 0.0

            # Useing 4 GTT
            total_labels = []
            total_pre_logits = []
            # Using 4 IR metrics
            total_labels_flat = np.array([])
            total_pre_logits_flat = np.array([])
            total_pre_scores_flat = np.array([])

            for x1, x2, x3, Y, sent_len, dia_len, tfs, pos_list in data_generator(data_name=data_name, mode=mode,
                                                                                  batch_size=batch_size,
                                                                                  nb_classes=nb_classes,
                                                                                  shuffle=shuffle, epoch=epoch):

                feed_dict = {self.input_x1: x1,
                             self.input_x2: x2,
                             self.input_x3: x3,
                             self.tfs: tfs,
                             self.pos_list: pos_list,
                             self.dia_len: dia_len,
                             self.sent_len: sent_len,
                             self.input_y: Y,
                             self.dropout_keep_prob: keep_prob}
                try:
                    _, step, loss, sequence, scores = self.train_op(feed_dict)
                    Y = np.argmax(Y, -1)
                    tmp_labels = np.array([])
                    tmp_pre_seq = np.array([])
                    tmp_pre_scores = np.array([])
                    for batch_id in range(len(dia_len)):
                        if batch_id == 0:
                            tmp_labels = Y[batch_id, :dia_len[batch_id]]
                            tmp_pre_seq = sequence[batch_id, :dia_len[batch_id]]
                            tmp_pre_scores = scores[batch_id, :dia_len[batch_id], 1]
                        else:
                            tmp_labels = np.concatenate([tmp_labels, Y[batch_id, :dia_len[batch_id]]])
                            tmp_pre_seq = np.concatenate([tmp_pre_seq, sequence[batch_id, :dia_len[batch_id]]])
                            tmp_pre_scores = np.concatenate([tmp_pre_scores, scores[batch_id, :dia_len[batch_id], 1]])

                        total_labels.append(Y[batch_id, :dia_len[batch_id]])
                        total_pre_logits.append(sequence[batch_id, :dia_len[batch_id]])

                    if counter == 0:
                        total_labels_flat = tmp_labels
                        total_pre_logits_flat = tmp_pre_seq
                        total_pre_scores_flat = tmp_pre_scores
                    else:
                        total_labels_flat = np.concatenate([total_labels_flat, tmp_labels], axis=0)
                        total_pre_logits_flat = np.concatenate([total_pre_logits_flat, tmp_pre_seq], axis=0)
                        total_pre_scores_flat = np.concatenate([total_pre_scores_flat, tmp_pre_scores], axis=0)

                    total_loss += loss
                    counter += 1

                except ValueError as e:
                    self.logger.info("Wrong batch.{}".format(e))

            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)

            total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
                get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)

            # calc AUC score
            fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
            auc_score = auc(fpr, tpr)

            print(confusion_matrix(total_labels_flat, total_pre_logits_flat))
            self.logger.info(
                "Handoff %s: Loss:%.4f\tAcc:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tAUC:%.4f\tGT-I:%.4f\tGT-II:%.4f\tGT-III:%.4f"
                % (mode, total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, auc_score, gtt_1,
                   gtt_2, gtt_3))
            self.logger.info("#############################################")

            if is_val:
                eval_loss, accuracy, f1score, macro_f1score, gt1, gt2, gt3 = \
                    self.evaluate_batch(data_generator, data_name, mode=val_mode,
                                        batch_size=batch_size, nb_classes=nb_classes, shuffle=False)

                metrics_dict = {"loss": eval_loss, "accuracy": accuracy, "macro_f1": macro_f1score,
                                "f1score": f1score, "auc": auc_score,
                                "gt1": gt1, "gt2": gt2, "gt3": gt3}

                if metrics_dict["f1score"] > max_val and save_best:
                    max_val = metrics_dict["f1score"]
                    self.save(self.save_dir, self.model_name + '.best')

        self.save(self.save_dir, self.model_name + '.last')

        if is_test:
            self.restore(self.save_dir, self.model_name + '.best')
            self.evaluate_batch(data_generator, data_name, mode=test_mode,
                                batch_size=batch_size, nb_classes=nb_classes, shuffle=False)

    def evaluate_batch(self, data_generator, data_name, mode='eval',
                       batch_size=20, nb_classes=2, shuffle=False):
        counter, total_loss = 0, 0.0

        # Useing 4 GTT
        total_labels = []
        total_pre_logits = []
        # Using 4 IR metrics
        total_labels_flat = np.array([])
        total_pre_logits_flat = np.array([])
        total_pre_scores_flat = np.array([])

        for x1, x2, x3, y, sent_len, dia_len, tfs, pos_list in data_generator(data_name=data_name, mode=mode,
                                                                              batch_size=batch_size,
                                                                              nb_classes=nb_classes, shuffle=shuffle):
            feed_dict = {self.input_x1: x1,
                         self.input_x2: x2,
                         self.input_x3: x3,
                         self.tfs: tfs,
                         self.pos_list: pos_list,
                         self.input_y: y,
                         self.sent_len: sent_len,
                         self.dia_len: dia_len,
                         self.dropout_keep_prob: 1.0}

            loss, sequence, scores = self.train_op(feed_dict)
            y = np.argmax(y, -1)
            tmp_labels = np.array([])
            tmp_pre_seq = np.array([])
            tmp_pre_scores = np.array([])
            for batch_id in range(len(dia_len)):
                if batch_id == 0:
                    tmp_labels = y[batch_id, :dia_len[batch_id]]
                    tmp_pre_seq = sequence[batch_id, :dia_len[batch_id]]
                    tmp_pre_scores = scores[batch_id, :dia_len[batch_id], 1]
                else:
                    tmp_labels = np.concatenate([tmp_labels, y[batch_id, :dia_len[batch_id]]])
                    tmp_pre_seq = np.concatenate([tmp_pre_seq, sequence[batch_id, :dia_len[batch_id]]])
                    tmp_pre_scores = np.concatenate([tmp_pre_scores, scores[batch_id, :dia_len[batch_id], 1]])

                total_labels.append(y[batch_id, :dia_len[batch_id]])
                total_pre_logits.append(sequence[batch_id, :dia_len[batch_id]])

            if counter == 0:
                total_labels_flat = tmp_labels
                total_pre_logits_flat = tmp_pre_seq
                total_pre_scores_flat = tmp_pre_scores
            else:
                total_labels_flat = np.concatenate([total_labels_flat, tmp_labels], axis=0)
                total_pre_logits_flat = np.concatenate([total_pre_logits_flat, tmp_pre_seq], axis=0)
                total_pre_scores_flat = np.concatenate([total_pre_scores_flat, tmp_pre_scores], axis=0)

            total_loss += loss
            counter += 1

        gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)

        total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
            get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)

        fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
        auc_score = auc(fpr, tpr)

        print(confusion_matrix(total_labels_flat, total_pre_logits_flat))
        self.logger.info(
            "Handoff %s: Loss:%.4f\tAcc:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tAUC:%.4f\tGT-I:%.4f\tGT-II:%.4f\tGT-III:%.4f"
            % (
            mode, total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, auc_score, gtt_1, gtt_2,
            gtt_3))
        if mode == 'test':
            self.logger.info(
                "Handoff point %s\tF1Score\tMacro_F1Score\tAUC\tGT-I\tGT-II\tGT-III")
            self.logger.info(
                "Metrics %s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"
                % (mode, total_f1_sent * 100, total_macro_sent * 100, auc_score * 100, gtt_1 * 100, gtt_2 * 100,
                   gtt_3 * 100))
            tmp_lambda = 0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))

            self.logger.info(classification_report(total_labels_flat, total_pre_logits_flat, digits=4))
            self.logger.info(confusion_matrix(total_labels_flat, total_pre_logits_flat))

        return total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, gtt_1, gtt_2, gtt_3

    def train_crf(self, data_generator, keep_prob, epochs, data_name,
                  mode='train', batch_size=20, nb_classes=2, shuffle=True,
                  is_val=True, is_test=True, save_best=True,
                  val_mode='eval', test_mode='test'):
        print('Using train crf trainer.')
        max_val = 0

        for epoch in range(epochs):
            self.logger.info('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))
            print('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))

            counter, total_loss = 0, 0.0

            # Useing 4 GTT
            total_labels = []
            total_pre_logits = []
            # Using 4 IR metrics
            total_labels_flat = np.array([])
            total_pre_logits_flat = np.array([])
            total_pre_scores_flat = np.array([])

            for x1, x2, x3, Y, sent_len, dia_len in data_generator(data_name=data_name, mode=mode,
                                                                   batch_size=batch_size, nb_classes=nb_classes,
                                                                   shuffle=shuffle, epoch=epoch):

                feed_dict = {self.input_x1: x1,
                             self.input_x2: x2,
                             self.input_x3: x3,
                             self.dia_len: dia_len,
                             self.sent_len: sent_len,
                             self.input_y: Y,
                             self.dropout_keep_prob: keep_prob}
                try:
                    _, step, loss, sequence, scores = self.train_op(feed_dict)
                    tmp_labels = np.array([])
                    tmp_pre_seq = np.array([])
                    tmp_pre_scores = np.array([])
                    for batch_id in range(len(dia_len)):
                        if batch_id == 0:
                            tmp_labels = Y[batch_id, :dia_len[batch_id]]
                            tmp_pre_seq = sequence[batch_id, :dia_len[batch_id]]
                            tmp_pre_scores = scores[batch_id, :dia_len[batch_id], 1]
                        else:
                            tmp_labels = np.concatenate([tmp_labels, Y[batch_id, :dia_len[batch_id]]])
                            tmp_pre_seq = np.concatenate([tmp_pre_seq, sequence[batch_id, :dia_len[batch_id]]])
                            tmp_pre_scores = np.concatenate([tmp_pre_scores, scores[batch_id, :dia_len[batch_id], 1]])

                        total_labels.append(Y[batch_id, :dia_len[batch_id]])
                        total_pre_logits.append(sequence[batch_id, :dia_len[batch_id]])

                    if counter == 0:
                        total_labels_flat = tmp_labels
                        total_pre_logits_flat = tmp_pre_seq
                        total_pre_scores_flat = tmp_pre_scores
                    else:
                        total_labels_flat = np.concatenate([total_labels_flat, tmp_labels], axis=0)
                        total_pre_logits_flat = np.concatenate([total_pre_logits_flat, tmp_pre_seq], axis=0)
                        total_pre_scores_flat = np.concatenate([total_pre_scores_flat, tmp_pre_scores], axis=0)

                    total_loss += loss
                    counter += 1

                except ValueError as e:
                    self.logger.info("Wrong batch.{}".format(e))

            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)

            total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
                get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)

            fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
            auc_score = auc(fpr, tpr)

            print(confusion_matrix(total_labels_flat, total_pre_logits_flat))
            self.logger.info(
                "Handoff %s: Loss:%.4f\tAcc:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tAUC:%.4f\tGT-I:%.4f\tGT-II:%.4f\tGT-III:%.4f"
                % (mode, total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, auc_score, gtt_1,
                   gtt_2, gtt_3))
            self.logger.info("#############################################")

            if is_val:
                eval_loss, accuracy, f1score, macro_f1score, gt1, gt2, gt3 = \
                    self.evaluate_batch_crf(data_generator, data_name, mode=val_mode,
                                            batch_size=batch_size, nb_classes=nb_classes, shuffle=False)

                metrics_dict = {"loss": eval_loss, "accuracy": accuracy, "macro_f1": macro_f1score,
                                "f1score": f1score, "auc": auc_score,
                                "gt1": gt1, "gt2": gt2, "gt3": gt3}
                if metrics_dict["f1score"] > max_val and save_best:
                    max_val = metrics_dict["f1score"]
                    self.save(self.save_dir, self.model_name + '.best')

        self.save(self.save_dir, self.model_name + '.last')

        if is_test:
            self.restore(self.save_dir, self.model_name + '.best')
            self.evaluate_batch_crf(data_generator, data_name, mode=test_mode,
                                    batch_size=batch_size, nb_classes=nb_classes, shuffle=False)

    def evaluate_batch_crf(self, data_generator, data_name, mode='eval',
                           batch_size=20, nb_classes=2, shuffle=False):
        counter, total_loss = 0, 0.0

        # Useing 4 GTT
        total_labels = []
        total_pre_logits = []
        # Using 4 IR metrics
        total_labels_flat = np.array([])
        total_pre_logits_flat = np.array([])
        total_pre_scores_flat = np.array([])

        for x1, x2, x3, y, sent_len, dia_len in data_generator(data_name=data_name, mode=mode, batch_size=batch_size,
                                                               nb_classes=nb_classes, shuffle=shuffle):
            feed_dict = {self.input_x1: x1,
                         self.input_x2: x2,
                         self.input_x3: x3,
                         self.input_y: y,
                         self.sent_len: sent_len,
                         self.dia_len: dia_len,
                         self.dropout_keep_prob: 1.0}

            loss, sequence, scores = self.train_op(feed_dict)

            tmp_labels = np.array([])
            tmp_pre_seq = np.array([])
            tmp_pre_scores = np.array([])
            for batch_id in range(len(dia_len)):
                if batch_id == 0:
                    tmp_labels = y[batch_id, :dia_len[batch_id]]
                    tmp_pre_seq = sequence[batch_id, :dia_len[batch_id]]
                    tmp_pre_scores = scores[batch_id, :dia_len[batch_id], 1]
                else:
                    tmp_labels = np.concatenate([tmp_labels, y[batch_id, :dia_len[batch_id]]])
                    tmp_pre_seq = np.concatenate([tmp_pre_seq, sequence[batch_id, :dia_len[batch_id]]])
                    tmp_pre_scores = np.concatenate([tmp_pre_scores, scores[batch_id, :dia_len[batch_id], 1]])

                total_labels.append(y[batch_id, :dia_len[batch_id]])
                total_pre_logits.append(sequence[batch_id, :dia_len[batch_id]])

            if counter == 0:
                total_labels_flat = tmp_labels
                total_pre_logits_flat = tmp_pre_seq
                total_pre_scores_flat = tmp_pre_scores
            else:
                total_labels_flat = np.concatenate([total_labels_flat, tmp_labels], axis=0)
                total_pre_logits_flat = np.concatenate([total_pre_logits_flat, tmp_pre_seq], axis=0)
                total_pre_scores_flat = np.concatenate([total_pre_scores_flat, tmp_pre_scores], axis=0)

            total_loss += loss
            counter += 1

        gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)

        total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
            get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)

        fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
        auc_score = auc(fpr, tpr)

        print(confusion_matrix(total_labels_flat, total_pre_logits_flat))
        self.logger.info(
            "Handoff %s: Loss:%.4f\tAcc:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tAUC:%.4f\tGT-I:%.4f\tGT-II:%.4f\tGT-III:%.4f"
            % (
            mode, total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, auc_score, gtt_1, gtt_2,
            gtt_3))
        if mode == 'test':
            self.logger.info(
                "Handoff point %s\tF1Score\tMacro_F1Score\tAUC\tGT-I\tGT-II\tGT-III")
            self.logger.info(
                "Metrics %s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"
                % (mode, total_f1_sent * 100, total_macro_sent * 100, auc_score * 100, gtt_1 * 100, gtt_2 * 100,
                   gtt_3 * 100))
            tmp_lambda = 0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            self.logger.info(classification_report(total_labels_flat, total_pre_logits_flat, digits=4))
            self.logger.info(confusion_matrix(total_labels_flat, total_pre_logits_flat))

        return total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, gtt_1, gtt_2, gtt_3

    def train_sup(self, data_generator, keep_prob, epochs, data_name,
                  mode='train', batch_size=20, nb_classes=2, shuffle=True,
                  is_val=True, is_test=True, save_best=True,
                  val_mode='eval', test_mode='test'):
        max_val = 0

        for epoch in range(epochs):
            self.logger.info('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))
            print('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))
            counter, total_loss = 0, 0.0

            # Useing 4 GTT
            total_labels = []
            total_pre_logits = []
            # Using 4 IR metrics
            total_labels_flat = np.array([])
            total_pre_logits_flat = np.array([])
            total_pre_scores_flat = np.array([])

            for x1, x2, x3, Y, sent_len, dia_len in data_generator(data_name=data_name, mode=mode,
                                                                   batch_size=batch_size, nb_classes=nb_classes,
                                                                   shuffle=shuffle, epoch=epoch):
                feed_dict = {self.input_x1: x1,
                             self.input_x2: x2,
                             self.input_x3: x3,
                             self.dia_len: dia_len,
                             self.sent_len: sent_len,
                             self.input_y: Y,
                             self.dropout_keep_prob: keep_prob}
                try:
                    _, step, loss, sequence, scores = self.train_op(feed_dict)
                    Y = np.argmax(Y, -1)
                    tmp_labels = np.array([])
                    tmp_pre_seq = np.array([])
                    tmp_pre_scores = np.array([])
                    for batch_id in range(len(dia_len)):
                        if batch_id == 0:
                            tmp_labels = Y[batch_id, :dia_len[batch_id]]
                            tmp_pre_seq = sequence[batch_id, :dia_len[batch_id]]
                            tmp_pre_scores = scores[batch_id, :dia_len[batch_id], 1]
                        else:
                            tmp_labels = np.concatenate([tmp_labels, Y[batch_id, :dia_len[batch_id]]])
                            tmp_pre_seq = np.concatenate([tmp_pre_seq, sequence[batch_id, :dia_len[batch_id]]])
                            tmp_pre_scores = np.concatenate([tmp_pre_scores, scores[batch_id, :dia_len[batch_id], 1]])

                        total_labels.append(Y[batch_id, :dia_len[batch_id]])
                        total_pre_logits.append(sequence[batch_id, :dia_len[batch_id]])

                    if counter == 0:
                        total_labels_flat = tmp_labels
                        total_pre_logits_flat = tmp_pre_seq
                        total_pre_scores_flat = tmp_pre_scores
                    else:
                        total_labels_flat = np.concatenate([total_labels_flat, tmp_labels], axis=0)
                        total_pre_logits_flat = np.concatenate([total_pre_logits_flat, tmp_pre_seq], axis=0)
                        total_pre_scores_flat = np.concatenate([total_pre_scores_flat, tmp_pre_scores], axis=0)

                    total_loss += loss
                    counter += 1

                except ValueError as e:
                    self.logger.info("Wrong batch.{}".format(e))

            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)

            total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
                get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)

            fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
            auc_score = auc(fpr, tpr)

            print(confusion_matrix(total_labels_flat, total_pre_logits_flat))
            self.logger.info(
                "Handoff %s: Loss:%.4f\tAcc:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tAUC:%.4f\tGT-I:%.4f\tGT-II:%.4f\tGT-III:%.4f"
                % (mode, total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, auc_score, gtt_1,
                   gtt_2, gtt_3))
            self.logger.info("#############################################")

            if is_val:
                eval_loss, accuracy, f1score, macro_f1score, gt1, gt2, gt3 = \
                    self.evaluate_batch_sup(data_generator, data_name, mode=val_mode,
                                            batch_size=batch_size, nb_classes=nb_classes, shuffle=False)

                metrics_dict = {"loss": eval_loss, "accuracy": accuracy, "macro_f1": macro_f1score,
                                "f1score": f1score, "auc": auc_score,
                                "gt1": gt1, "gt2": gt2, "gt3": gt3}

                if metrics_dict["f1score"] > max_val and save_best:
                    max_val = metrics_dict["f1score"]
                    self.save(self.save_dir, self.model_name + '.best')

        self.save(self.save_dir, self.model_name + '.last')

        if is_test:
            self.restore(self.save_dir, self.model_name + '.best')
            self.evaluate_batch_sup(data_generator, data_name, mode=test_mode,
                                    batch_size=batch_size, nb_classes=nb_classes, shuffle=False)

    def evaluate_batch_sup(self, data_generator, data_name, mode='eval',
                           batch_size=20, nb_classes=2, shuffle=False):
        counter, total_loss = 0, 0.0

        # Useing 4 GTT
        total_labels = []
        total_pre_logits = []
        # Using 4 IR metrics
        total_labels_flat = np.array([])
        total_pre_logits_flat = np.array([])
        total_pre_scores_flat = np.array([])

        for x1, x2, x3, y, sent_len, dia_len, ids in data_generator(data_name=data_name, mode=mode,
                                                                    batch_size=batch_size, nb_classes=nb_classes,
                                                                    shuffle=shuffle):
            feed_dict = {self.input_x1: x1,
                         self.input_x2: x2,
                         self.input_x3: x3,
                         self.input_y: y,
                         self.sent_len: sent_len,
                         self.dia_len: dia_len,
                         self.dropout_keep_prob: 1.0}

            loss, sequence, scores = self.train_op(feed_dict)
            y = np.argmax(y, -1)
            tmp_labels = np.array([])
            tmp_pre_seq = np.array([])
            tmp_pre_scores = np.array([])
            for batch_id in range(len(dia_len)):
                if batch_id == 0:
                    tmp_labels = y[batch_id, :dia_len[batch_id]]
                    tmp_pre_seq = sequence[batch_id, :dia_len[batch_id]]
                    tmp_pre_scores = scores[batch_id, :dia_len[batch_id], 1]
                else:
                    tmp_labels = np.concatenate([tmp_labels, y[batch_id, :dia_len[batch_id]]])
                    tmp_pre_seq = np.concatenate([tmp_pre_seq, sequence[batch_id, :dia_len[batch_id]]])
                    tmp_pre_scores = np.concatenate([tmp_pre_scores, scores[batch_id, :dia_len[batch_id], 1]])

                total_labels.append(y[batch_id, :dia_len[batch_id]])
                total_pre_logits.append(sequence[batch_id, :dia_len[batch_id]])

            if counter == 0:
                total_labels_flat = tmp_labels
                total_pre_logits_flat = tmp_pre_seq
                total_pre_scores_flat = tmp_pre_scores
            else:
                total_labels_flat = np.concatenate([total_labels_flat, tmp_labels], axis=0)
                total_pre_logits_flat = np.concatenate([total_pre_logits_flat, tmp_pre_seq], axis=0)
                total_pre_scores_flat = np.concatenate([total_pre_scores_flat, tmp_pre_scores], axis=0)

            total_loss += loss
            counter += 1

        gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)

        total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
            get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)

        fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
        auc_score = auc(fpr, tpr)

        print(confusion_matrix(total_labels_flat, total_pre_logits_flat))
        self.logger.info(
            "Handoff %s: Loss:%.4f\tAcc:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tAUC:%.4f\tGT-I:%.4f\tGT-II:%.4f\tGT-III:%.4f"
            % (
            mode, total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, auc_score, gtt_1, gtt_2,
            gtt_3))
        if mode == 'test':
            self.logger.info(
                "Handoff point %s\tF1Score\tMacro_F1Score\tAUC\tGT-I\tGT-II\tGT-III")
            self.logger.info(
                "Metrics %s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"
                % (mode, total_f1_sent * 100, total_macro_sent * 100, auc_score * 100, gtt_1 * 100, gtt_2 * 100,
                   gtt_3 * 100))
            tmp_lambda = 0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = 0.
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.25
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.5
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.75
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            tmp_lambda = -0.99
            gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
            self.logger.info("Lambda={}\t{}\t{}\t{}".format(tmp_lambda, gtt_1, gtt_2, gtt_3))
            self.logger.info(classification_report(total_labels_flat, total_pre_logits_flat))
            self.logger.info(confusion_matrix(total_labels_flat, total_pre_logits_flat))

        return total_loss / float(counter), total_acc_sent, total_f1_sent, total_macro_sent, gtt_1, gtt_2, gtt_3

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(model_dir, model_prefix)
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        print("Restore path: {}".format(str(os.path.join(model_dir, model_prefix))))
        self.saver.restore(model_dir, model_prefix)
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))


if __name__ == '__main__':

    logger = logging.getLogger("Tensorflow")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' ')[:3])

    network = Network()

    log_path = './networks/logs/' + network.model_name + '.log'
    if os.path.exists(log_path):
        os.remove(log_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    network.build_graph()
