#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: multivariate_normal_distribution.py
Author: Wan Li
Date: 2018/12/7 16:14:36
"""

import numpy as np
import tensorflow as tf

class MultivariateNormalDistribution(object):
    """
        Multivariate Normal Distribution Sampling
    """
    def __init__(self):
        """
            Initializer
        """
        self._build_graph()
        self._init_session()
    
    def _construct_placeholders(self):
        """
            Construct input placeholders
        """
        self.mu = tf.placeholder(shape=[None], dtype=tf.float32, name='mu')
        self.cov = tf.placeholder(shape=[None], dtype=tf.float32, name='cov')
    
    def _build_graph(self):
        """
            Build graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._construct_placeholders()
            mnd = tf.contrib.distributions.MultivariateNormalFullCovariance(
                loc=self.mu,
                covariance_matrix = tf.reshape(
                    self.cov,
                    tf.concat([tf.shape(self.cov)[:-1], [tf.shape(self.mu)[-1], tf.shape(self.mu)[-1]]],
                             axis = 0)))
            self.samples = mnd.sample()
            self.init_var_op = tf.global_variables_initializer()
    
    def _init_session(self):
        """
            Init session
        """
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_var_op)
    
    def sample(self, mu, cov):
        """
            Inference
        """
        return self.session.run(self.samples, feed_dict={
            self.mu: np.array(mu),
            self.cov: np.array(cov)
        })

    def export(self, export_dir):
        """
            Export model
        """
        with self.session as sess:
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map= {
                    "serving_default": tf.saved_model.signature_def_utils.build_signature_def(
                        inputs= {
                            "mu": tf.saved_model.utils.build_tensor_info(self.mu),
                            "cov": tf.saved_model.utils.build_tensor_info(self.cov)
                        },
                        outputs= {"samples": tf.saved_model.utils.build_tensor_info(self.samples)},
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                    })
            builder.save(as_text=False)

if __name__ == "__main__":
    import shutil
    import os
    DIR = "./model"
    if os.path.exists(DIR):
        shutil.rmtree(DIR)

    b = MultivariateNormalDistribution()
    for _ in range(10):
        print(b.sample(
            [1.9702128918649626, 1.1395280221264217, -0.627184572276027],
            [0.00560475, -0.00378602, -0.00205141, -0.00378602, 0.00716149, -0.00093772, -0.00205141, -0.00093772, 0.00787278]))
    b.export(DIR)
    
    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], DIR)
        signature = meta_graph_def.signature_def
        a_tensor_name = signature["serving_default"].inputs["mu"].name
        b_tensor_name = signature["serving_default"].inputs["cov"].name
        s_tensor_name = signature["serving_default"].outputs["samples"].name
        a_tensor = sess.graph.get_tensor_by_name(a_tensor_name)
        b_tensor = sess.graph.get_tensor_by_name(b_tensor_name)
        s_tensor = sess.graph.get_tensor_by_name(s_tensor_name)
        for _ in range(10):
            print(sess.run(s_tensor, feed_dict={
                a_tensor: [1.9702128918649626, 1.1395280221264217, -0.627184572276027],
                b_tensor: [0.00560475, -0.00378602, -0.00205141, -0.00378602, 0.00716149, -0.00093772, -0.00205141, -0.00093772, 0.00787278]
            }))
