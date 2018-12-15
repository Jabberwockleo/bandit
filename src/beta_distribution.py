#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: beta_distribution.py
Author: Wan Li
Date: 2018/12/7 16:14:36
"""

import numpy as np
import tensorflow as tf

class BetaDistribution(object):
    """
        Beta Distribution Sampling
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
        self.alpha = tf.placeholder(shape=[None], dtype=tf.float32, name='alpha')
        self.beta = tf.placeholder(shape=[None], dtype=tf.float32, name='beta')
    
    def _build_graph(self):
        """
            Build graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._construct_placeholders()
            bd = tf.distributions.Beta(self.alpha, self.beta)
            self.samples = bd.sample()
            self.init_var_op = tf.global_variables_initializer()
    
    def _init_session(self):
        """
            Init session
        """
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_var_op)
    
    def sample(self, alphas, betas):
        """
            Inference
        """
        return self.session.run(self.samples, feed_dict={
            self.alpha: np.array(alphas),
            self.beta: np.array(betas)
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
                    "model": tf.saved_model.signature_def_utils.build_signature_def(
                        inputs= {"alpha": tf.saved_model.utils.build_tensor_info(self.alpha)},
                        outputs= {"beta": tf.saved_model.utils.build_tensor_info(self.beta)})
                    })
            builder.save(as_text=False)

if __name__ == "__main__":
    import shutil
    import os
    DIR = "./model"
    if os.path.exists(DIR):
        shutil.rmtree(DIR)

    b = BetaDistribution()
    for _ in range(10):
        print(b.sample([1, 2, 22], [1, 9, 1]))
    b.export(DIR)
