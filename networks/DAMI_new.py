#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras.initializers import GlorotNormal
from utility import *
from networks.layers.attention import *
from networks.layers.transformer import *
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

class DAMI(tf.keras.Model):
    def __init__(self, memory=0, vocab=None, config_dict=None, log_path=None, **kwargs):
        super(DAMI, self).__init__()
        self.model_name = self.__class__.__name__
        self.logger = logging.getLogger("Tensorflow")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Model Name: {self.model_name}")

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Configuration parameters
        self.rnn_dim = config_dict["rnn_dim"]
        self.dense_dim = config_dict["dense_dim"]
        self.pos_dim = config_dict["pos2id"]
        self.lr = config_dict["learning_rate"]
        self.l2_reg_lambda = config_dict["l2_reg_lambda"]
        self.weight_decay = config_dict["weight_decay"]
        self.clip_norm = 1.0  # Add gradient clipping threshold
        self.lamb = 0.  # Add missing lamb parameter for GTT scoring
        
        # Data parameters
        self.dia_max_len = 30
        self.sent_max_len = 50
        self.nb_classes = 2
        self.embedding_dim = 200
        self.nb_words = None
        self.data_name = None
        self.name = None
        
        # Initialize layers
        self.embedding_layer = None  # Will be initialized when nb_words is set
        if vocab and vocab.embeddings is not None:
            # Make sure embeddings match the vocabulary size
            self.embeddings = vocab.embeddings[:vocab.size()]
        else:
            self.embeddings = None
        
        self.dropout = Dropout(0.5)
        
        # Attention layers with proper initialization
        self.attention_agent = layers.Dense(
            2 * self.rnn_dim,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6)  # Add small L2 regularization
        )
        self.attention_customer = layers.Dense(
            2 * self.rnn_dim,
            activation='tanh',
            kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(1e-6)  # Add small L2 regularization
        )
        
        # RNN layers with proper initialization
        self.bidirectional_rnn = Bidirectional(
            LSTM(
                self.rnn_dim,
                return_sequences=True,
                return_state=True,
                kernel_initializer='glorot_normal',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                name='bidirectional_lstm'
            ),
            name='bidirectional_wrapper'
        )
        self.dialogue_rnn = LSTM(
            self.rnn_dim,
            return_sequences=True,
            kernel_initializer='glorot_normal',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            name='dialogue_lstm'
        )
        
        # Dense layers with proper initialization
        self.local_inference = Dense(
            self.dense_dim,
            activation='relu',
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.context_aware = Dense(
            self.dense_dim,
            activation='relu',
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.output_layer = Dense(
            self.nb_classes,
            activation='softmax',
            kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        
        # Context-aware attention weight matrix
        self.w_datt = tf.Variable(
            tf.random.normal(shape=[self.rnn_dim, self.rnn_dim], dtype=tf.float16, stddev=0.01),  # Use smaller stddev
            trainable=True,
            name='w_datt'
        )
        
        # Optimizer with gradient clipping
        self.optimizer = optimizers.Adam(
            learning_rate=self.lr * 0.05,  # Increase learning rate slightly for better progress
            clipnorm=self.clip_norm,
            epsilon=1e-8  # Increase epsilon for better numerical stability
        )

    def set_nb_words(self, nb_words):
        """Set the vocabulary size and initialize the embedding layer"""
        self.nb_words = nb_words
        
        # Ensure embeddings match the vocabulary size if they exist
        if self.embeddings is not None:
            if len(self.embeddings) > self.nb_words:
                self.embeddings = self.embeddings[:self.nb_words]
            elif len(self.embeddings) < self.nb_words:
                # Pad with zeros if needed
                padding = np.zeros((self.nb_words - len(self.embeddings), self.embedding_dim))
                self.embeddings = np.vstack([self.embeddings, padding])
        
        self.embedding_layer = layers.Embedding(
            input_dim=self.nb_words,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.constant_initializer(self.embeddings) if self.embeddings is not None else GlorotNormal(),
            trainable=True
        )

    def set_data_name(self, data_name):
        """Set the data name"""
        self.data_name = data_name

    def set_name(self, name):
        """Set the model name"""
        self.name = name

    def set_from_model_config(self, model_config):
        """Set parameters from model config"""
        # Already handled in __init__
        pass

    def set_from_data_config(self, data_config):
        """Set parameters from data config"""
        if 'nb_classes' in data_config:
            self.nb_classes = data_config['nb_classes']
            # Reinitialize output layer with new number of classes
            self.output_layer = Dense(self.nb_classes, activation='softmax')

    def build_graph(self):
        """Build the model graph - no-op in Keras, as graph is built dynamically"""
        pass

    def call(self, inputs, training=False):
        # Unpack inputs dictionary
        input_x1 = inputs['input_x1']
        input_x2 = inputs['input_x2']
        input_x3 = inputs['input_x3']
        tfs = inputs['tfs']
        pos_list = inputs['pos_list']
        sent_len = inputs['sent_len']
        dia_len = inputs['dia_len']
        
        # Convert input_x1 to int32 for embedding lookup
        input_x1_int = tf.cast(input_x1, tf.int32)
        # Embedding layer
        embedded = self.embedding_layer(input_x1_int)
        
        # Check for NaN in embeddings
        if tf.reduce_any(tf.math.is_nan(embedded)):
            self.logger.warning("NaN detected in embeddings, fixing...")
            embedded = tf.where(tf.math.is_nan(embedded), tf.zeros_like(embedded), embedded)
        
        if training:
            embedded = self.dropout(embedded, training=training)
            tfs = self.dropout(tfs, training=training)
        
        # Get batch size and reshape inputs
        batch_size = tf.shape(input_x1)[0]
        
        # Get input dtype for consistent casting
        input_dtype = embedded.dtype
        
        # Reshape embedded from [batch_size, dia_max_len, sent_max_len, embedding_dim] 
        # to [batch_size * dia_max_len, sent_max_len, embedding_dim]
        embedded_reshaped = tf.reshape(embedded, shape=[-1, self.sent_max_len, self.embedding_dim])
        
        # Reshape tfs and add channel dimension
        tfs_reshaped = tf.expand_dims(tf.reshape(tfs, shape=[-1, self.sent_max_len]), axis=-1)
        
        # For pos_list: [batch_size, dia_max_len, sent_max_len, pos_dim] 
        # to [batch_size * dia_max_len, sent_max_len, pos_dim]
        pos_list_reshaped = tf.cast(
            tf.reshape(pos_list, shape=[batch_size * self.dia_max_len, self.sent_max_len, self.pos_dim]),
            input_dtype
        )
        
        # Reshape sentence lengths
        sent_len_reshape = tf.reshape(sent_len, shape=[-1])
        
        # Positional encoding
        sent_pos_emb = tf.cast(positional_encoding(embedded_reshaped, self.sent_max_len), input_dtype)
        
        # Ensure all tensors have the same dtype before concatenation
        embedded_reshaped = tf.cast(embedded_reshaped, input_dtype)
        
        # Concatenate along the last axis
        embedded_all = tf.concat([embedded_reshaped, sent_pos_emb, pos_list_reshaped], axis=-1)
        
        # Check for NaN after concatenation
        if tf.reduce_any(tf.math.is_nan(embedded_all)):
            self.logger.warning("NaN detected after concatenation, fixing...")
            embedded_all = tf.where(tf.math.is_nan(embedded_all), tf.zeros_like(embedded_all), embedded_all)
        
        # Sentence encoding
        sent_encoder_output = self.bidirectional_rnn(sequences=embedded_all, training=training)
        
        # Debug: Print shapes to understand the issue
        self.logger.info(f"Bidirectional RNN output type: {type(sent_encoder_output)}")
        if isinstance(sent_encoder_output, tuple):
            self.logger.info(f"Bidirectional RNN output tuple length: {len(sent_encoder_output)}")
            for i, item in enumerate(sent_encoder_output):
                if hasattr(item, 'shape'):
                    self.logger.info(f"Tuple item {i} shape: {item.shape}")
        
        # Check for NaN after bidirectional RNN - handle tuple output properly
        if isinstance(sent_encoder_output, tuple):
            # Unpack the tuple: (output, forward_h, forward_c, backward_h, backward_c)
            # output shape: [batch_size * dia_max_len, sent_max_len, 2 * rnn_dim]
            # forward_h, backward_h shape: [batch_size * dia_max_len, rnn_dim]
            output, forward_h, forward_c, backward_h, backward_c = sent_encoder_output
            
            # Check for NaN in the output sequence
            if tf.reduce_any(tf.math.is_nan(output)):
                self.logger.warning("NaN detected after bidirectional RNN output, fixing...")
                output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
            
            # Use the hidden states (h) for the final state
            # Concatenate forward and backward hidden states
            sent_encoder_state = tf.concat([forward_h, backward_h], axis=-1)  # Shape: [batch_size * dia_max_len, 2 * rnn_dim]
            sent_encoder_output = output  # Shape: [batch_size * dia_max_len, sent_max_len, 2 * rnn_dim]
        else:
            # If return_state is False, we just get the output sequence
            if tf.reduce_any(tf.math.is_nan(sent_encoder_output)):
                self.logger.warning("NaN detected after bidirectional RNN, fixing...")
                sent_encoder_output = tf.where(tf.math.is_nan(sent_encoder_output), tf.zeros_like(sent_encoder_output), sent_encoder_output)
            # Take the last timestep as the state
            sent_encoder_state = sent_encoder_output[:, -1, :]  # Shape: [batch_size * dia_max_len, 2 * rnn_dim]
        
        # Check for NaN in encoder state
        if tf.reduce_any(tf.math.is_nan(sent_encoder_state)):
            self.logger.warning("NaN detected in encoder state, fixing...")
            sent_encoder_state = tf.where(tf.math.is_nan(sent_encoder_state), tf.zeros_like(sent_encoder_state), sent_encoder_state)
        
        # Attention mechanism
        agent_tag = tf.reshape(input_x2, shape=[-1])
        customer_tag = 1 - agent_tag
        
        # Agent attention
        qk_a = self.attention_agent(embedded_all)  # This will have input_dtype
        
        # Check for NaN in attention weights
        if tf.reduce_any(tf.math.is_nan(qk_a)):
            self.logger.warning("NaN detected in agent attention weights")
            # Instead of returning safe values, try to fix the NaN
            qk_a = tf.where(tf.math.is_nan(qk_a), tf.zeros_like(qk_a), qk_a)
        
        # Cast tfs_reshaped to match qk_a's dtype
        tfs_reshaped_cast = tf.cast(tfs_reshaped, input_dtype)
        vu_a = tf.nn.relu(tf.reduce_sum(qk_a * (1 - tfs_reshaped_cast), axis=-1))
        dif_mask = tf.sequence_mask(sent_len_reshape, maxlen=self.sent_max_len, dtype=input_dtype)
        sent_paddings = tf.ones_like(dif_mask, dtype=input_dtype) * tf.cast(-2**32+1, input_dtype)
        vu_masked_a = tf.where(tf.equal(dif_mask, 0), sent_paddings, tf.cast(vu_a, input_dtype))
        
        # Expand agent_tag to match vu_masked_a dimensions
        agent_tag_expanded = tf.expand_dims(agent_tag, axis=-1)  # Shape: [batch_size * dia_max_len, 1]
        agent_tag_expanded = tf.tile(agent_tag_expanded, [1, self.sent_max_len])  # Shape: [batch_size * dia_max_len, sent_max_len]
        vu_masked_a = tf.where(tf.equal(agent_tag_expanded, 0), sent_paddings, vu_masked_a)
        
        # Add numerical stability to softmax
        vu_masked_a = vu_masked_a - tf.reduce_max(vu_masked_a, axis=-1, keepdims=True)
        alphas_a = tf.nn.softmax(vu_masked_a)
        
        # Customer attention
        qk_c = self.attention_customer(embedded_all)  # This will have input_dtype
        
        # Check for NaN in customer attention weights
        if tf.reduce_any(tf.math.is_nan(qk_c)):
            self.logger.warning("NaN detected in customer attention weights")
            # Instead of returning safe values, try to fix the NaN
            qk_c = tf.where(tf.math.is_nan(qk_c), tf.zeros_like(qk_c), qk_c)
        
        vu_c = tf.reduce_sum(qk_c * (1 - tfs_reshaped_cast), axis=-1)
        vu_masked_c = tf.where(tf.equal(dif_mask, 0), sent_paddings, tf.cast(vu_c, input_dtype))
        
        # Expand customer_tag to match vu_masked_c dimensions
        customer_tag_expanded = tf.expand_dims(customer_tag, axis=-1)  # Shape: [batch_size * dia_max_len, 1]
        customer_tag_expanded = tf.tile(customer_tag_expanded, [1, self.sent_max_len])  # Shape: [batch_size * dia_max_len, sent_max_len]
        vu_masked_c = tf.where(tf.equal(customer_tag_expanded, 0), sent_paddings, vu_masked_c)
        
        # Add numerical stability to softmax
        vu_masked_c = vu_masked_c - tf.reduce_max(vu_masked_c, axis=-1, keepdims=True)
        alphas_c = tf.nn.softmax(vu_masked_c)
        
        # Combine attention
        alphas = alphas_a + alphas_c
        sent_encoder_output_atten = tf.reduce_sum(sent_encoder_output * tf.expand_dims(alphas, -1), 1)
        
        # Check for NaN after attention and fix if needed
        if tf.reduce_any(tf.math.is_nan(sent_encoder_output_atten)):
            self.logger.warning("NaN detected after attention mechanism, fixing...")
            sent_encoder_output_atten = tf.where(tf.math.is_nan(sent_encoder_output_atten), 
                                               tf.zeros_like(sent_encoder_output_atten), 
                                               sent_encoder_output_atten)
        
        # Reshape for dialogue processing
        sent_encoder_attened_reshape = tf.reshape(sent_encoder_output_atten, shape=[-1, self.dia_max_len, 2 * self.rnn_dim])
        sent_encoder_state_reshape = tf.reshape(sent_encoder_state, shape=[-1, self.dia_max_len, 2 * self.rnn_dim])
        
        # Combine embeddings - ensure all tensors have the same dtype
        senti_score = tf.cast(tf.expand_dims(input_x3, axis=-1), input_dtype)
        sent_encoder_attened_reshape = tf.cast(sent_encoder_attened_reshape, input_dtype)
        sent_encoder_state_reshape = tf.cast(sent_encoder_state_reshape, input_dtype)
        
        combine_emb = tf.concat([sent_encoder_attened_reshape, sent_encoder_state_reshape, senti_score], axis=-1)
        if training:
            combine_emb = self.dropout(combine_emb, training=training)
        
        # Local inference
        cross_match = tf.matmul(combine_emb, tf.transpose(combine_emb, perm=[0, 2, 1]))
        cross_match_upper = tf.linalg.band_part(cross_match, num_lower=0, num_upper=-1)
        cross_match_sim_one_direct = cross_match - cross_match_upper
        encode_local = tf.concat([combine_emb, cross_match_sim_one_direct], axis=-1)
        encode_local = self.local_inference(encode_local)
        if training:
            encode_local = self.dropout(encode_local, training=training)
        
        # Dialogue encoding
        v_dia_encode_output = self.dialogue_rnn(sequences=encode_local, training=training)
        
        # Context-aware attention
        dia_seq_mask = tf.sequence_mask(dia_len, maxlen=self.dia_max_len, dtype=input_dtype)
        
        # Ensure consistent data types for matrix multiplication
        v_dia_encode_output_cast = tf.cast(v_dia_encode_output, input_dtype)
        w_datt_cast = tf.cast(self.w_datt, input_dtype)
        
        d_att = tf.matmul(tf.matmul(v_dia_encode_output_cast, w_datt_cast), tf.transpose(v_dia_encode_output_cast, perm=[0, 2, 1]))
        d_att_mask = tf.tile(tf.expand_dims(dia_seq_mask, axis=-1), multiples=[1, 1, self.dia_max_len])
        paddings = tf.ones_like(d_att, dtype=input_dtype) * tf.cast(-2**32+1, input_dtype)
        a_att_masked = tf.where(tf.equal(d_att_mask, 0), paddings, d_att)
        
        # Add numerical stability to softmax
        a_att_masked = a_att_masked - tf.reduce_max(a_att_masked, axis=-1, keepdims=True)
        d_att_alpha = tf.nn.softmax(tf.linalg.band_part(a_att_masked, num_lower=-1, num_upper=0))
        
        # Ensure consistent data types for matrix multiplication
        v_dia_encode_output_cast_for_context = tf.cast(v_dia_encode_output, input_dtype)
        context_encode = tf.matmul(d_att_alpha, v_dia_encode_output_cast_for_context)
        concat_d = tf.concat([context_encode, v_dia_encode_output], axis=-1)
        combine_h = self.context_aware(concat_d)
        if training:
            combine_h = self.dropout(combine_h, training=training)
        
        # Output layer
        logits = self.output_layer(combine_h)
        
        # Check for NaN in final logits
        if tf.reduce_any(tf.math.is_nan(logits)):
            self.logger.warning("NaN detected in final logits, fixing...")
            logits = tf.where(tf.math.is_nan(logits), tf.zeros_like(logits), logits)
        
        # Add numerical stability to final softmax
        logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        output = tf.argmax(logits, axis=-1)
        proba = tf.nn.softmax(logits)
        
        # Final check for NaN in probabilities
        if tf.reduce_any(tf.math.is_nan(proba)):
            self.logger.warning("NaN detected in final probabilities, fixing...")
            proba = tf.where(tf.math.is_nan(proba), tf.ones_like(proba) / self.nb_classes, proba)
        
        return logits, output, proba

    def train_step(self, data):
        batch_x1, batch_x2, batch_x3, batch_tf, batch_pos, batch_labels, batch_sent_len, batch_dia_len = data
        
        # Convert dia_len to tensor if it's not already
        if not isinstance(batch_dia_len, tf.Tensor):
            batch_dia_len = tf.convert_to_tensor(batch_dia_len, dtype=tf.int32)
        
        # Get the target dtype from the embedding layer
        target_dtype = self.embedding_layer.dtype
        
        # Create inputs dictionary with all tensors cast to appropriate types
        inputs = {
            'input_x1': tf.cast(batch_x1, target_dtype),  # For embedding lookup
            'input_x2': tf.cast(batch_x2, target_dtype),  # Agent/customer tags
            'input_x3': tf.cast(batch_x3, target_dtype),  # Sentiment scores
            'tfs': tf.cast(batch_tf, target_dtype),       # Text features
            'pos_list': tf.cast(batch_pos, tf.int32),     # Position list stays as int32
            'sent_len': tf.cast(batch_sent_len, tf.int32),  # Lengths stay as int32
            'dia_len': batch_dia_len                      # Already int32
        }
        
        with tf.GradientTape() as tape:
            logits, _, _ = self.call(inputs, training=True)
            # Reshape logits and labels to match
            logits_flat = tf.reshape(logits, [-1, self.nb_classes])
            labels_flat = tf.reshape(batch_labels, [-1, self.nb_classes])
            # Cast labels to match logits dtype
            labels_flat = tf.cast(labels_flat, logits_flat.dtype)
            
            # Check for NaN values in logits
            if tf.reduce_any(tf.math.is_nan(logits_flat)):
                self.logger.warning("NaN values detected in logits")
                return {'loss': float('inf')}
            
            # Add numerical stability to softmax
            logits_stable = logits_flat - tf.reduce_max(logits_flat, axis=-1, keepdims=True)
            
            # Compute cross entropy loss with numerical stability
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    labels_flat,
                    tf.nn.softmax(logits_stable),
                    from_logits=True
                )
            )
            
            if self.l2_reg_lambda > 0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables])
                loss += self.l2_reg_lambda * l2_loss
        
        # Check for NaN values in loss
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            self.logger.warning(f"Invalid loss detected in training step: {loss}")
            return {'loss': float('inf')}
        
        # Compute and clip gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Check for NaN values in gradients
        if any(tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None):
            self.logger.warning("NaN gradients detected in training step")
            return {'loss': float('inf')}
        
        # Check for infinite gradients
        if any(tf.reduce_any(tf.math.is_inf(g)) for g in gradients if g is not None):
            self.logger.warning("Infinite gradients detected in training step")
            return {'loss': float('inf')}
        
        # Filter out None gradients
        valid_gradients = [(g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None]
        
        if not valid_gradients:
            self.logger.warning("No valid gradients found")
            return {'loss': float('inf')}
        
        # Apply clipped gradients
        self.optimizer.apply_gradients(valid_gradients)
        
        return {'loss': loss}

    def test_step(self, data):
        batch_x1, batch_x2, batch_x3, batch_tf, batch_pos, batch_labels, batch_sent_len, batch_dia_len = data
        
        # Convert dia_len to tensor if it's not already
        if not isinstance(batch_dia_len, tf.Tensor):
            batch_dia_len = tf.convert_to_tensor(batch_dia_len, dtype=tf.int32)
        
        # Get the target dtype from the embedding layer
        target_dtype = self.embedding_layer.dtype
        
        # Create inputs dictionary with all tensors cast to appropriate types
        inputs = {
            'input_x1': tf.cast(batch_x1, target_dtype),  # For embedding lookup
            'input_x2': tf.cast(batch_x2, target_dtype),  # Agent/customer tags
            'input_x3': tf.cast(batch_x3, target_dtype),  # Sentiment scores
            'tfs': tf.cast(batch_tf, target_dtype),       # Text features
            'pos_list': tf.cast(batch_pos, tf.int32),     # Position list stays as int32
            'sent_len': tf.cast(batch_sent_len, tf.int32),  # Lengths stay as int32
            'dia_len': batch_dia_len                      # Already int32
        }
        
        logits, _, _ = self.call(inputs, training=False)
        # Reshape logits and labels to match
        logits_flat = tf.reshape(logits, [-1, self.nb_classes])
        labels_flat = tf.reshape(batch_labels, [-1, self.nb_classes])
        # Cast labels to match logits dtype
        labels_flat = tf.cast(labels_flat, logits_flat.dtype)
        
        # Check for NaN values in logits
        if tf.reduce_any(tf.math.is_nan(logits_flat)):
            self.logger.warning("NaN values detected in logits during test")
            return {'loss': float('inf')}
        
        # Add numerical stability to softmax
        logits_stable = logits_flat - tf.reduce_max(logits_flat, axis=-1, keepdims=True)
        
        # Compute cross entropy loss with numerical stability
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels_flat,
                tf.nn.softmax(logits_stable),
                from_logits=True
            )
        )
        
        # Check for NaN values
        if tf.math.is_nan(loss):
            self.logger.warning("NaN loss detected in test step")
            return {'loss': float('inf')}
        
        return {'loss': loss}

    def train(self, data_generator, keep_prob, epochs, data_name, mode='train',
             batch_size=20, nb_classes=2, shuffle=True, is_val=True, is_test=True, save_best=True):
        """Custom training loop that maintains compatibility with the old interface"""
        self.logger.info("Starting training...")
        best_val_metric = float('-inf')  # Changed from best_val_loss since we're tracking f1score
        best_epoch = 0
        patience = 5  # Early stopping patience
        no_improvement_count = 0
        min_loss_threshold = 1e-6  # Minimum loss threshold to consider training stuck
        max_epochs_without_progress = 10  # Maximum epochs without any progress
        
        # Create checkpoint manager for model saving
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint_dir = f'./checkpoints/{self.model_name}_{data_name}'
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        
        for epoch in range(epochs):
            # Training
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            train_loss = 0
            train_batches = 0
            nan_batch_count = 0  # Count batches with NaN loss
            epoch_start_time = time.time()
            
            # Get training data
            train_data = data_generator(data_name, mode, batch_size, nb_classes)
            
            for batch_idx, batch_data in enumerate(train_data):
                try:
                    # Add timeout check for individual batches
                    batch_start_time = time.time()
                    metrics = self.train_step(batch_data)
                    batch_loss = metrics['loss']
                    
                    # Check if batch took too long (more than 30 seconds)
                    if time.time() - batch_start_time > 30:
                        self.logger.warning(f"Batch {batch_idx} took too long, skipping...")
                        continue
                    
                    # Check for infinite or NaN loss
                    if tf.math.is_inf(batch_loss) or tf.math.is_nan(batch_loss):
                        nan_batch_count += 1
                        self.logger.warning(f"Batch {batch_idx}: Invalid loss detected ({batch_loss}), skipping...")
                        continue
                    
                    train_loss += batch_loss
                    train_batches += 1
                    
                    # Add progress logging every 5 batches
                    if batch_idx % 5 == 0:
                        self.logger.info(f"Batch {batch_idx}: Loss = {batch_loss:.4f}")
                        
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    nan_batch_count += 1
                    continue
                
                # Safety check: if too many NaN batches, stop training
                if nan_batch_count > 10:
                    self.logger.error("Too many NaN batches detected, stopping training")
                    return
                
                # Safety check: if epoch is taking too long, stop
                if time.time() - epoch_start_time > 300:  # 5 minutes per epoch
                    self.logger.error("Epoch taking too long, stopping training")
                    return
            
            # Calculate average loss
            if train_batches > 0:
                avg_train_loss = train_loss / train_batches
            else:
                self.logger.error("No valid batches processed, stopping training")
                return
            
            self.logger.info(f"Training loss: {avg_train_loss:.4f} (valid batches: {train_batches})")
            
            # Check if loss is stuck
            if avg_train_loss < min_loss_threshold:
                self.logger.warning(f"Loss too low ({avg_train_loss:.6f}), training might be stuck")
            
            # Validation
            if is_val:
                try:
                    val_metrics = self.evaluate(data_generator, data_name, mode='eval', 
                                             batch_size=batch_size, nb_classes=nb_classes)
                    self.logger.info(f"Validation metrics: {val_metrics}")
                    
                    # Save best model based on F1 score
                    if save_best and val_metrics['f1score'] > best_val_metric:
                        best_val_metric = val_metrics['f1score']
                        best_epoch = epoch + 1
                        no_improvement_count = 0
                        save_path = manager.save()
                        self.logger.info(f"Saved best model at epoch {best_epoch} to {save_path}")
                    else:
                        no_improvement_count += 1
                        
                    # Early stopping
                    if no_improvement_count >= patience:
                        self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                        break
                        
                    # Safety check: if no progress for too many epochs
                    if no_improvement_count >= max_epochs_without_progress:
                        self.logger.error(f"No progress for {max_epochs_without_progress} epochs, stopping training")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error during validation: {e}")
                    no_improvement_count += 1
            
            # Testing
            if is_test:
                try:
                    test_metrics = self.evaluate(data_generator, data_name, mode='test',
                                              batch_size=batch_size, nb_classes=nb_classes)
                    self.logger.info(f"Test metrics: {test_metrics}")
                except Exception as e:
                    self.logger.error(f"Error during testing: {e}")
        
        if save_best:
            self.logger.info(f"Best model was from epoch {best_epoch} with validation F1 score: {best_val_metric:.4f}")
            # Restore best model
            best_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if best_ckpt:
                checkpoint.restore(best_ckpt)
                self.logger.info(f"Restored best model from {best_ckpt}")

    def evaluate(self, data_generator, data_name, mode='test', batch_size=20, nb_classes=2):
        """Evaluate the model on a specific dataset"""
        self.logger.info(f"Evaluating on {data_name} {mode} set...")
        total_loss = 0
        total_batches = 0
        
        # For GTT metrics
        total_labels = []
        total_pre_logits = []
        
        # For other metrics
        total_labels_flat = []
        total_pre_logits_flat = []
        total_pre_scores_flat = []
        
        # Get evaluation data
        eval_data = data_generator(data_name, mode, batch_size, nb_classes)
        
        for batch_data in eval_data:
            # Get predictions
            inputs = {
                'input_x1': tf.cast(batch_data[0], self.embedding_layer.dtype),
                'input_x2': tf.cast(batch_data[1], self.embedding_layer.dtype),
                'input_x3': tf.cast(batch_data[2], self.embedding_layer.dtype),
                'tfs': tf.cast(batch_data[3], self.embedding_layer.dtype),
                'pos_list': tf.cast(batch_data[4], tf.int32),
                'sent_len': tf.cast(batch_data[6], tf.int32),
                'dia_len': tf.cast(batch_data[7], tf.int32)
            }
            
            logits, predictions, probabilities = self.call(inputs, training=False)
            labels = batch_data[5]
            
            # Compute metrics
            metrics = self.test_step(batch_data)
            total_loss += metrics['loss']
            total_batches += 1
            
            # Process each batch for metrics
            dia_len = batch_data[7]
            labels_np = np.argmax(labels, axis=-1)  # Convert one-hot to class indices
            
            for batch_id in range(len(dia_len)):
                # For GTT metrics
                total_labels.append(labels_np[batch_id, :dia_len[batch_id]])
                total_pre_logits.append(predictions[batch_id, :dia_len[batch_id]])
                
                # For flat metrics
                total_labels_flat.extend(labels_np[batch_id, :dia_len[batch_id]])
                total_pre_logits_flat.extend(predictions[batch_id, :dia_len[batch_id]])
                total_pre_scores_flat.extend(probabilities[batch_id, :dia_len[batch_id], 1])
    
        # Convert lists to numpy arrays
        total_labels_flat = np.array(total_labels_flat)
        total_pre_logits_flat = np.array(total_pre_logits_flat)
        total_pre_scores_flat = np.array(total_pre_scores_flat)
        
        # Check for NaN values in the arrays
        if np.any(np.isnan(total_pre_scores_flat)):
            self.logger.warning("NaN values detected in prediction scores, using safe defaults")
            total_pre_scores_flat = np.zeros_like(total_pre_scores_flat)
        
        # Compute average loss
        avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
        
        # Compute GTT scores
        gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=self.lamb)
        
        # Compute other metrics
        total_acc_sent, total_p_sent, total_r_sent, total_f1_sent, total_macro_sent, _, _ = \
            get_a_p_r_f_sara(target=total_labels_flat, predict=total_pre_logits_flat, category=1)
        
        # Compute ROC AUC with NaN handling
        try:
            fpr, tpr, thresholds = roc_curve(total_labels_flat, total_pre_scores_flat, pos_label=1)
            auc_score = auc(fpr, tpr)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error computing ROC AUC: {e}, using default value")
            auc_score = 0.5  # Default to random classifier
        
        # Log detailed metrics
        self.logger.info(
            f"Handoff {mode}: Loss:{avg_loss:.4f}\tAcc:{total_acc_sent:.4f}\t"
            f"F1Score:{total_f1_sent:.4f}\tMacro_F1Score:{total_macro_sent:.4f}\t"
            f"AUC:{auc_score:.4f}\tGT-I:{gtt_1:.4f}\tGT-II:{gtt_2:.4f}\tGT-III:{gtt_3:.4f}"
        )
        
        if mode == 'test':
            # Log additional test metrics
            self.logger.info("Handoff point %s\tF1Score\tMacro_F1Score\tAUC\tGT-I\tGT-II\tGT-III")
            self.logger.info(
                f"Metrics {mode}\t{total_f1_sent*100:.2f}\t{total_macro_sent*100:.2f}\t"
                f"{auc_score*100:.2f}\t{gtt_1*100:.2f}\t{gtt_2*100:.2f}\t{gtt_3*100:.2f}"
            )
            
            # Test different lambda values for GTT scores
            for tmp_lambda in [0.99, 0.75, 0.5, 0.25, 0., -0.25, -0.5, -0.75, -0.99]:
                gtt_1, gtt_2, gtt_3 = get_gtt_score(total_labels, total_pre_logits, lamb=tmp_lambda)
                self.logger.info(f"Lambda={tmp_lambda}\t{gtt_1}\t{gtt_2}\t{gtt_3}")
            
            # Log classification report and confusion matrix
            self.logger.info("\nClassification Report:")
            self.logger.info(classification_report(total_labels_flat, total_pre_logits_flat, digits=4))
            self.logger.info("\nConfusion Matrix:")
            self.logger.info(confusion_matrix(total_labels_flat, total_pre_logits_flat))
        
        return {
            'loss': avg_loss,
            'accuracy': total_acc_sent,
            'precision': total_p_sent,
            'recall': total_r_sent,
            'f1score': total_f1_sent,
            'macro_f1': total_macro_sent,
            'auc': auc_score,
            'gt1': gtt_1,
            'gt2': gtt_2,
            'gt3': gtt_3,
            'predictions': total_pre_logits_flat,
            'labels': total_labels_flat
        }

    def save(self, filepath):
        """Save the model weights"""
        self.save_weights(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model weights"""
        self.load_weights(filepath)
        self.logger.info(f"Model loaded from {filepath}") 