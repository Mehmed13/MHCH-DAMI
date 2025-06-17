#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
@Reference: https://www.jianshu.com/p/cc6407444a8c   <Hierarchical attention networks for document classification.>
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : attention.py
@Time     : 19-1-19 下午1:38
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import warnings
import tensorflow as tf
from tensorflow.keras import layers

warnings.filterwarnings('ignore', category=FutureWarning)

class AttentionLayer(layers.Layer):
    """Custom attention layer for RNN outputs"""
    
    def __init__(self, attention_size, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_size = attention_size
        self.w_omega = None
        self.b_omega = None
        self.u_omega = None
        
    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.w_omega = self.add_weight(
            name='w_omega',
            shape=(hidden_size, self.attention_size),
            initializer='glorot_normal',
            trainable=True
        )
        self.b_omega = self.add_weight(
            name='b_omega',
            shape=(self.attention_size,),
            initializer='zeros',
            trainable=True
        )
        self.u_omega = self.add_weight(
            name='u_omega',
            shape=(self.attention_size,),
            initializer='glorot_normal',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs, time_major=False, return_alphas=False):
        """
        Implement attention mechanism for RNN layer.
        
        Args:
            inputs: Tensor. RNN outputs.
            time_major: Boolean. If True, the inputs dimension is (T, B, D)
            return_alphas: Boolean. If True, return the alphas attention weights.
            
        Returns:
            Attention-ed outputs and alphas(Optional).
        """
        if isinstance(inputs, tuple):
            # In case of Bi-RNNs not concatenate the forward and the backward RNN outputs
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T, B, D) => (B, T, D)
            inputs = tf.transpose(inputs, [1, 0, 2])

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps
        v = tf.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)

        # For each of the timestamps its vector of size A from 'v' is reduced with 'u' vector
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # (B, T) shape
        alphas = tf.nn.softmax(vu, name='alphas')

        # Outputs of (Bi)RNNs is reduced with attention vector
        outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return outputs
        else:
            return outputs, alphas
            
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'attention_size': self.attention_size
        })
        return config

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Legacy function for backward compatibility.
    Use AttentionLayer class for new code.
    """
    layer = AttentionLayer(attention_size)
    return layer(inputs, time_major=time_major, return_alphas=return_alphas)
