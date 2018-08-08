import math

import models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

import scipy.io as sio
import numpy as np

import utils
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "ensemble_moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LinearRegressionModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, original_input=None, **unused_params):
    """Creates a linear regression model.
    Args:
      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    num_methods = model_input.get_shape().as_list()[-1]
    weight = tf.get_variable("ensemble_weight", 
        shape=[num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight = tf.nn.softmax(weight)
    output = tf.einsum("ijk,k->ij", model_input, weight)
    return {"predictions": output}
  
  
class MeanModel(models.BaseModel):
  """Mean model."""

  def create_model(self, model_input, **unused_params):
    """Creates a logistic model.
      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = tf.reduce_mean(model_input, axis=2)
    return {"predictions": output}
  
class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None, 
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.
     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.
    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]

    flat_input = tf.reshape(model_input, shape=[-1,num_features * num_methods])

    tensor_weight = tf.get_variable("tensor_weight",
        shape=[num_features, num_methods, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    tensor_bias = tf.get_variable("tensor_bias",
        shape=[num_features, num_methods],
        initializer=tf.zeros_initializer(),
        regularizer=slim.l2_regularizer(l2_penalty))

    gate_activations = tf.einsum("ijk,jkl->ijl", model_input, tensor_weight) \
        + tf.expand_dims(tensor_bias, dim=0)

    output = tf.reduce_sum(model_input * tf.nn.softmax(gate_activations), axis=2)
    return {"predictions": output}
  
class NonunitMatrixRegressionModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, original_input=None, epsilon=1e-5, **unused_params):
    """Creates a non-unified matrix regression model.
    Args:
      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""

    num_features = model_input.get_shape().as_list()[-2]
    num_methods = model_input.get_shape().as_list()[-1]

    log_model_input = tf.stop_gradient(tf.log((epsilon + model_input) / (1.0 + epsilon - model_input)))
    
    weight = tf.get_variable("ensemble_weight", 
        shape=[num_features, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight = tf.nn.softmax(weight)

    output = tf.nn.sigmoid(tf.einsum("ijk,jk->ij", log_model_input, weight))
    return {"predictions": output}
  
class MatrixRegressionModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, original_input=None, **unused_params):
    """Creates a matrix regression model.
    Args:
      model_input: 'batch' x 'num_features' x 'num_methods' matrix of input features.
      vocab_size: The number of classes in the dataset.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    num_features = model_input.get_shape().as_list()[-2]
    num_methods = model_input.get_shape().as_list()[-1]

    weight1d = tf.get_variable("ensemble_weight1d", 
        shape=[num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight2d = tf.get_variable("ensemble_weight2d", 
        shape=[num_features, num_methods],
        regularizer=slim.l2_regularizer(10 * l2_penalty))
    weight = tf.nn.softmax(tf.einsum("ij,j->ij", weight2d, weight1d), dim=-1)
    output = tf.einsum("ijk,jk->ij", model_input, weight)
    return {"predictions": output}
