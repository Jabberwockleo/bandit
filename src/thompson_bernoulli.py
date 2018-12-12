#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2018 Wan Li. All Rights Reserved
#
########################################################################

"""
File: thompson_bernoulli.py
Author: leowan(leowan)
Date: 2018/12/7 16:14:36

Reference:
    A Tutorial on Thompson Sampling
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf

    Beta Distribution
    https://en.wikipedia.org/wiki/Beta_distribution
"""

import json
from collections import defaultdict

import numpy as np
import pymc as pm

KEY_PUBLISH = 'publish'
KEY_CLICK = 'click'

class ThompsonSamplingBernoulli(object):
    """
        Thompson Sampling for Bernoulli Bandit
    """
    def __init__(self, action_names):
        """
            Initializer
            Params:
                action_names: array of action names, string type
        """
        self.action_names = action_names
        self.action_record = defaultdict(dict)
        # Setup prior distribution
        for an in self.action_names:
            self.action_record[an] = {
                KEY_PUBLISH: 2,
                KEY_CLICK: 1
            }

    def choose(self):
        """
            Make decision
        """
        clicks = []
        publishs = []
        skips = []
        for i in range(len(self.action_names)):
            clicks.append(self.action_record[self.action_names[i]][KEY_CLICK])
            publishs.append(self.action_record[self.action_names[i]][KEY_PUBLISH])
        clicks = np.array(clicks)
        publishs = np.array(publishs)
        skips = publishs - clicks
        probs = []
        for click, skip in zip(clicks, skips):
            probs.append(pm.rbeta(click, skip))
        probs = np.array(probs)
        print(probs)
        index = np.argmax(probs)
        return self.action_names[index]

    def record_publish(self, action_name):
        """
            Record publish
        """
        self.action_record[action_name][KEY_PUBLISH] += 1

    def record_click(self, action_name):
        """
            Record publish
        """
        self.action_record[action_name][KEY_CLICK] += 1

    def deserialize(self, serialized):
        """
            Deserialize
        """
        self.action_record = json.loads(serialized)
        self.action_names = self.action_record.keys()

    def serialize(self):
        """
            Serialize
        """
        return json.dumps(self.action_record)

if __name__ == "__main__":
    bandit = ThompsonSamplingBernoulli(['action_a', 'action_b', 'action_c'])

    chosen = bandit.choose()
    print('chosen: {}'.format(chosen))
    bandit.record_publish(chosen)

    chosen = bandit.choose()
    print('chosen: {}'.format(chosen))
    bandit.record_publish(chosen)
    bandit.record_click(chosen)

    serialized = bandit.serialize()
    print('serialized: {}'.format(serialized))

    new_bandit = ThompsonSamplingBernoulli([])
    new_bandit.deserialize(serialized)
    chosen = bandit.choose()
    print('new chosen: {}'.format(chosen))
    bandit.record_publish(chosen)
    serialized = bandit.serialize()
    print('new serialized: {}'.format(serialized))
