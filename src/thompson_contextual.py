#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: thompson.py
Author: leowan(leowan)
Date: 2018/12/7 16:14:36

Reference:
    Thompson Sampling for Contextual Bandits with Linear Payoffs (ICML 2013)
    http://proceedings.mlr.press/v28/agrawal13.pdf

    An Empirical Evaluation of Thompson Sampling (NIPS 2011)
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/thompson.pdf

    Sub-Gaussian distribution
    https://en.wikipedia.org/wiki/Sub-Gaussian_distribution
"""

import json
import numpy as np

class ThompsonSamplingContextual(object):
    """
        Thompson Sampling for Contextual Bandit
    """
    def __init__(self, context_dimension, R=0.01, delta=0.5, epsilon=0.3):
        """
            Initializer
            Params:
                context_dimension: number of context(feature) dimensions
                R: float [0, +inf), according to the assumption that Pr(reward) - b.T\mu is R-sub-Gaussian distribution
                delta: float (0, 1), regret bound is satisfied with probability of 1 - delta
                epsilon: float (0, 1), a parameter for Thompson sampling. If total trials T is known, epsilon = 1/ln(T)
        """
        self.context_dimension = context_dimension
        self.R = R
        self.delta = delta
        self.epsilon = epsilon
        
        self.B = np.identity(self.context_dimension)
        self.f = np.zeros(shape=(self.context_dimension, 1))
        self.mu_hat = np.zeros(shape=(self.context_dimension, 1))
        self.v = self.R * np.sqrt(24 / self.epsilon * self.context_dimension * np.log(1 / self.delta))
        self.sigma_squared_hat = self.v**2 * np.linalg.pinv(self.B)
        
    def _inference(self, context):
        """
            Sample a possible reward from reward distribution
            Params:
                context: feature vector
            Return:
                index of chosen context(feature vector for item)
        """
        context = np.array(context)
        mu_tilde = np.random.multivariate_normal(
            self.mu_hat.flat,
            self.sigma_squared_hat)[..., np.newaxis]
        expected_reward = context.dot(self.mu_hat)
        dynamic_reward = context.dot(mu_tilde)
        return expected_reward, dynamic_reward
    
    def _reward(self, context, reward):
        """
            Update model by observation
            Params:
                context: feature vector
                reward: float, observed reward score
        """
        context = np.array(context).reshape([-1, 1])
        self.B += context.dot(context.T)
        self.f += reward * context
        self.mu_hat = np.linalg.pinv(self.B).dot(self.f)
        self.sigma_squared_hat = self.v**2 * np.linalg.pinv(self.B)

    def choose(self, contexts):
        """
            Make decision
            Params:
                contexts: array of feature vectors
            Return:
                index of chosen context(feature vector for item)
        """
        predict_rewards = [self._inference(c)[1] for c in contexts]
        chosen_index = np.argmax(predict_rewards)
        print("predicts: {}, choose: {}".format(predict_rewards, chosen_index))
        return chosen_index

    def feedback(self, contexts, rewards):
        """
            Receive reward and update model
            Params:
                contexts: array of feature vectors
                rewards: array of float
        """
        for c, r in zip(contexts, rewards):
            self._reward(c, r)

    def deserialize(self, serialized):
        """
            Deserialize
        """
        model = json.loads(serialized)
        self.R = model['R']
        self.delta = model['delta']
        self.epsilon = model['epsilon']
        self.context_dimension = model['dims']
        self.B = np.array(model['B'])
        self.mu_hat = np.array(model['mu_hat'])
        self.f = np.array(model['f'])
        self.v = self.R * np.sqrt(24 / self.epsilon * self.context_dimension * np.log(1 / self.delta))
        self.mu_hat = np.linalg.pinv(self.B).dot(self.f)
        self.sigma_squared_hat = self.v**2 * np.linalg.pinv(self.B)

    def serialize(self):
        """
            Serialize
        """
        model = {
            'B': self.B.tolist(),
            'mu_hat': self.mu_hat.tolist(),
            'f': self.f.tolist(),
            'R': self.R,
            'delta': self.delta,
            'epsilon': self.epsilon,
            'dims': self.context_dimension
        }
        return json.dumps(model)

if __name__ == "__main__":
    a = ThompsonSamplingContextual(context_dimension=3)
    b = ThompsonSamplingContextual(context_dimension=3)
    b.deserialize(a.serialize())
    assert b.serialize() == a.serialize()

    def ground_truth(x):
        """
            Generate test data
        """
        assert len(x) == 3
        return x[0] * 3 + x[1] * 1 - x[2] * 2

    for _ in range(10):
        xx = np.random.uniform(0, 1, 3)
        b.feedback([xx], [ground_truth(xx)])

    print(b.mu_hat)
    
    import matplotlib.pyplot as plt
    plt.hist([b._inference([1, 0, 0])[1][0] for i in range(1000)], alpha=0.5, bins=50)
    plt.hist([b._inference([0, 1, 0])[1][0] for i in range(1000)], alpha=0.5, bins=50)
    plt.hist([b._inference([0, 0, 1])[1][0] for i in range(1000)], alpha=0.5, bins=50)
    plt.show()