# !/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

#root relative squared error
def rse(label, pred):
    """computes the root relative squared error
    (condensed using standard deviation formula)
    """
    #compute the root of the sum of the squared error
    numerator = np.sqrt(np.mean(np.square(label - pred), axis = None))
    #numerator = np.sqrt(np.sum(np.square(label - pred), axis=None))

    #compute the RMSE if we were to simply predict the average of the previous values
    denominator = np.std(label, axis = None)
    #denominator = np.sqrt(np.sum(np.square(label - np.mean(label, axis = None)), axis=None))

    return numerator / denominator

_rse = mx.metric.create(rse)

#relative absolute error
def rae(label, pred):
    """computes the relative absolute error
    (condensed using standard deviation formula)"""

    #compute the root of the sum of the squared error
    numerator = np.mean(np.abs(label - pred), axis=None)
    #numerator = np.sum(np.abs(label - pred), axis = None)

    #compute AE if we were to simply predict the average of the previous values
    denominator = np.mean(np.abs(label - np.mean(label, axis=None)), axis=None)
    #denominator = np.sum(np.abs(label - np.mean(label, axis = None)), axis=None)

    return numerator / denominator

_rae = mx.metric.create(rae)

#empirical correlation coefficient
def corr(label, pred):
    """computes the empirical correlation coefficient"""

    #compute the root of the sum of the squared error
    numerator1 = label - np.mean(label, axis=0)
    numerator2 = pred - np.mean(pred, axis = 0)
    numerator = np.mean(numerator1 * numerator2, axis=0)

    #compute the root of the sum of the squared error if we were to simply predict the average of the previous values
    denominator = np.std(label, axis=0) * np.std(pred, axis=0)

    #value passed here should be 321 numbers
    return np.mean(numerator / denominator)

_corr = mx.metric.create(corr)

#use mxnet native metric function
def get_custom_metrics():
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [_rse, _rae, _corr]:
        eval_metrics.add(child_metric)
    return eval_metrics


#create a composite metric manually as a sanity check whilst training
def metrics(label, pred):
    return ["RSE: ", rse(label, pred), "RAE: ", rae(label, pred), "CORR: ", corr(label, pred)]