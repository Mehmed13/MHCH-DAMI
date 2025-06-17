# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class LayerNormalization(layers.Layer):
    """Layer normalization layer"""
    
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        
    def build(self, input_shape):
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True
        )
        super(LayerNormalization, self).build(input_shape)
        
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** 0.5)
        return self.gamma * normalized + self.beta

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    embeddings = tf.Variable(
        tf.random.normal([vocab_size, num_units], stddev=0.1),
        name='weight_mat'
    )
    if zero_pad:
        embeddings = tf.concat([
            tf.zeros([1, num_units]),
            embeddings[1:, :]
        ], 0)
    return embeddings

def scaled_dot_product_attention(Q, K, V, key_masks,
                               causality=False, dropout_rate=0.,
                               training=True):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    '''
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # key masking
    outputs = mask(outputs, key_masks=key_masks, type="key")

    # causality or future blinding masking
    if causality:
        outputs = mask(outputs, type="future")

    # softmax
    outputs = tf.nn.softmax(outputs)
    attention = tf.transpose(outputs, [0, 2, 1])

    # dropout
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs, training=training)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.cast(key_masks, tf.float32)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])
        key_masks = tf.expand_dims(key_masks, 1)
        outputs = inputs + key_masks * padding_num
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")
        outputs = inputs

    return outputs

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer"""
    
    def __init__(self, num_heads=8, dropout_rate=0, causality=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        
    def build(self, input_shape):
        self.d_model = input_shape[-1]
        self.dense_q = layers.Dense(self.d_model)
        self.dense_k = layers.Dense(self.d_model)
        self.dense_v = layers.Dense(self.d_model)
        self.dropout = layers.Dropout(self.dropout_rate)
        super(MultiHeadAttention, self).build(input_shape)
        
    def call(self, queries, keys, values, key_masks, training=True):
        # Linear projections
        Q = self.dense_q(queries)  # (N, T_q, d_model)
        K = self.dense_k(keys)  # (N, T_k, d_model)
        V = self.dense_v(values)  # (N, T_k, d_model)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(
            Q_, K_, V_, key_masks, self.causality, self.dropout_rate, training
        )

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, d_model)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = LayerNormalization()(outputs)
 
        return outputs

def positional_encoding(inputs, maxlen, masking=True):
    """Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    """
    # Get input dtype and convert to float32 for computation
    input_dtype = inputs.dtype
    inputs = tf.cast(inputs, tf.float32)
    
    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

    # lookup
    outputs = tf.nn.embedding_lookup(position_enc, position_ind)

    # masks
    if masking:
        outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

    # Convert back to input dtype
    return tf.cast(outputs, input_dtype)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    """Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
    until it reaches init_lr.
    """
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)