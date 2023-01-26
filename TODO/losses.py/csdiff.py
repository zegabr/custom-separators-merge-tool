# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-classes-have-attributes
"""Built-in loss functions."""

import abc
import functools

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@keras_export('keras.losses.Loss')
class Loss:
81796$
  """Loss base class.

81796$
  To be implemented by subclasses:
81796$
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

81796$
  Example subclass implementation:

81796$
  ```python
81796$
  class MeanSquaredError(Loss):

81796$
    def call(self, y_true, y_pred):
81796$
      y_pred = tf.convert_to_tensor_v2(y_pred)
81796$
      y_true = tf.cast(y_true, y_pred.dtype)
81796$
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
81796$
  ```

81796$
  When used with `tf.distribute.Strategy`, outside of built-in training loops
81796$
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
81796$
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
81796$
  'SUM_OVER_BATCH_SIZE' will raise an error.

81796$
  Please see this custom training [tutorial](
81796$
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
81796$
  details on this.

81796$
  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:

81796$
  ```python
81796$
  with strategy.scope():
81796$
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
81796$
        reduction=tf.keras.losses.Reduction.NONE)
81796$
    ....
81796$
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
81796$
            (1. / global_batch_size))
81796$
  ```
81796$
  """

81796$
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
81796$
    """Initializes `Loss` class.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op.
81796$
    """
81796$
    losses_utils.ReductionV2.validate(reduction)
81796$
    self.reduction = reduction
81796$
    self.name = name
81796$
    # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
81796$
    # CannedEstimators.
81796$
    self._allow_sum_over_batch_size = False
81796$
    self._set_name_scope()

81796$
  def _set_name_scope(self):
81796$
    """Creates a valid `name_scope` name."""
81796$
    if self.name is None:
81796$
      self._name_scope = self.__class__.__name__
81796$
    elif self.name == '<lambda>':
81796$
      self._name_scope = 'lambda'
81796$
    else:
81796$
      # E.g. '_my_loss' => 'my_loss'
81796$
      self._name_scope = self.name.strip('_')

81796$
  def __call__(self, y_true, y_pred, sample_weight=None):
81796$
    """Invokes the `Loss` instance.

81796$
    Args:
81796$
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
81796$
        sparse loss functions such as sparse categorical crossentropy where
81796$
        shape = `[batch_size, d0, .. dN-1]`
81796$
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
81796$
      sample_weight: Optional `sample_weight` acts as a coefficient for the
81796$
        loss. If a scalar is provided, then the loss is simply scaled by the
81796$
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
81796$
        the total loss for each sample of the batch is rescaled by the
81796$
        corresponding element in the `sample_weight` vector. If the shape of
81796$
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
81796$
        this shape), then each loss element of `y_pred` is scaled
81796$
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
81796$
          functions reduce by 1 dimension, usually axis=-1.)

81796$
    Returns:
81796$
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
81796$
        shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
81796$
        because all loss functions reduce by 1 dimension, usually axis=-1.)

81796$
    Raises:
81796$
      ValueError: If the shape of `sample_weight` is invalid.
81796$
    """
81796$
    # If we are wrapping a lambda function strip '<>' from the name as it is not
81796$
    # accepted in scope name.
81796$
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
81796$
        y_true, y_pred, sample_weight)
81796$
    with backend.name_scope(self._name_scope), graph_ctx:
81796$
      if context.executing_eagerly():
81796$
        call_fn = self.call
81796$
      else:
81796$
        call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
81796$
      losses = call_fn(y_true, y_pred)
81796$
      return losses_utils.compute_weighted_loss(
81796$
          losses, sample_weight, reduction=self._get_reduction())

81796$
  @classmethod
81796$
  def from_config(cls, config):
81796$
    """Instantiates a `Loss` from its config (output of `get_config()`).

81796$
    Args:
81796$
        config: Output of `get_config()`.

81796$
    Returns:
81796$
        A `Loss` instance.
81796$
    """
81796$
    return cls(**config)

81796$
  def get_config(self):
81796$
    """Returns the config dictionary for a `Loss` instance."""
81796$
    return {'reduction': self.reduction, 'name': self.name}

81796$
  @abc.abstractmethod
81796$
  @doc_controls.for_subclass_implementers
81796$
  def call(self, y_true, y_pred):
81796$
    """Invokes the `Loss` instance.

81796$
    Args:
81796$
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
81796$
        sparse loss functions such as sparse categorical crossentropy where
81796$
        shape = `[batch_size, d0, .. dN-1]`
81796$
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

81796$
    Returns:
81796$
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
81796$
    """
81796$
    raise NotImplementedError('Must be implemented in subclasses.')

81796$
  def _get_reduction(self):
81796$
    """Handles `AUTO` reduction cases and returns the reduction value."""
81796$
    if (not self._allow_sum_over_batch_size and
81796$
        distribution_strategy_context.has_strategy() and
81796$
        (self.reduction == losses_utils.ReductionV2.AUTO or
81796$
         self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)):
81796$
      raise ValueError(
81796$
          'Please use `tf.keras.losses.Reduction.SUM` or '
81796$
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
81796$
          'used with `tf.distribute.Strategy` outside of the built-in training '
81796$
          'loops. You can implement '
81796$
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
81796$
          'size like:\n```\nwith strategy.scope():\n'
81796$
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
81796$
          'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
81796$
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
81796$
          '(1. / global_batch_size)\n```\nPlease see '
81796$
          'https://www.tensorflow.org/tutorials/distribute/custom_training'
81796$
          ' for more details.')

81796$
    if self.reduction == losses_utils.ReductionV2.AUTO:
81796$
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
81796$
    return self.reduction


class LossFunctionWrapper(Loss):
81796$
  """Wraps a loss function in the `Loss` class."""

81796$
  def __init__(self,
81796$
               fn,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name=None,
81796$
               **kwargs):
81796$
    """Initializes `LossFunctionWrapper` class.

81796$
    Args:
81796$
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
81796$
        **kwargs)`.
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: (Optional) name for the loss.
81796$
      **kwargs: The keyword arguments that are passed on to `fn`.
81796$
    """
81796$
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
81796$
    self.fn = fn
81796$
    self._fn_kwargs = kwargs

81796$
  def call(self, y_true, y_pred):
81796$
    """Invokes the `LossFunctionWrapper` instance.

81796$
    Args:
81796$
      y_true: Ground truth values.
81796$
      y_pred: The predicted values.

81796$
    Returns:
81796$
      Loss values per sample.
81796$
    """
81796$
    if tensor_util.is_tf_type(y_pred) and tensor_util.is_tf_type(y_true):
81796$
      y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

81796$
    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
81796$
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

81796$
  def get_config(self):
81796$
    config = {}
81796$
    for k, v in self._fn_kwargs.items():
81796$
      config[k] = backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
81796$
    base_config = super(LossFunctionWrapper, self).get_config()
81796$
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.losses.MeanSquaredError')
class MeanSquaredError(LossFunctionWrapper):
81796$
  """Computes the mean of squares of errors between labels and predictions.

81796$
  `loss = square(y_true - y_pred)`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[1., 1.], [1., 0.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> mse = tf.keras.losses.MeanSquaredError()
81796$
  >>> mse(y_true, y_pred).numpy()
81796$
  0.5

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
81796$
  0.25

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> mse = tf.keras.losses.MeanSquaredError(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> mse(y_true, y_pred).numpy()
81796$
  1.0

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> mse = tf.keras.losses.MeanSquaredError(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> mse(y_true, y_pred).numpy()
81796$
  array([0.5, 0.5], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='mean_squared_error'):
81796$
    """Initializes `MeanSquaredError` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'mean_squared_error'.
81796$
    """
81796$
    super(MeanSquaredError, self).__init__(
81796$
        mean_squared_error, name=name, reduction=reduction)


@keras_export('keras.losses.MeanAbsoluteError')
class MeanAbsoluteError(LossFunctionWrapper):
81796$
  """Computes the mean of absolute difference between labels and predictions.

81796$
  `loss = abs(y_true - y_pred)`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[1., 1.], [1., 0.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> mae = tf.keras.losses.MeanAbsoluteError()
81796$
  >>> mae(y_true, y_pred).numpy()
81796$
  0.5

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> mae(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
81796$
  0.25

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> mae = tf.keras.losses.MeanAbsoluteError(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> mae(y_true, y_pred).numpy()
81796$
  1.0

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> mae = tf.keras.losses.MeanAbsoluteError(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> mae(y_true, y_pred).numpy()
81796$
  array([0.5, 0.5], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanAbsoluteError())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='mean_absolute_error'):
81796$
    """Initializes `MeanAbsoluteError` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'mean_absolute_error'.
81796$
    """
81796$
    super(MeanAbsoluteError, self).__init__(
81796$
        mean_absolute_error, name=name, reduction=reduction)


@keras_export('keras.losses.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(LossFunctionWrapper):
81796$
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

81796$
  `loss = 100 * abs(y_true - y_pred) / y_true`

81796$
  Standalone usage:

81796$
  >>> y_true = [[2., 1.], [2., 3.]]
81796$
  >>> y_pred = [[1., 1.], [1., 0.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError()
81796$
  >>> mape(y_true, y_pred).numpy()
81796$
  50.

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> mape(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
81796$
  20.

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> mape(y_true, y_pred).numpy()
81796$
  100.

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> mape(y_true, y_pred).numpy()
81796$
  array([25., 75.], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd',
81796$
                loss=tf.keras.losses.MeanAbsolutePercentageError())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='mean_absolute_percentage_error'):
81796$
    """Initializes `MeanAbsolutePercentageError` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to
81796$
        'mean_absolute_percentage_error'.
81796$
    """
81796$
    super(MeanAbsolutePercentageError, self).__init__(
81796$
        mean_absolute_percentage_error, name=name, reduction=reduction)


@keras_export('keras.losses.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(LossFunctionWrapper):
81796$
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

81796$
  `loss = square(log(y_true + 1.) - log(y_pred + 1.))`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[1., 1.], [1., 0.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError()
81796$
  >>> msle(y_true, y_pred).numpy()
81796$
  0.240

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> msle(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
81796$
  0.120

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> msle(y_true, y_pred).numpy()
81796$
  0.480

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> msle(y_true, y_pred).numpy()
81796$
  array([0.240, 0.240], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd',
81796$
                loss=tf.keras.losses.MeanSquaredLogarithmicError())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='mean_squared_logarithmic_error'):
81796$
    """Initializes `MeanSquaredLogarithmicError` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to
81796$
        'mean_squared_logarithmic_error'.
81796$
    """
81796$
    super(MeanSquaredLogarithmicError, self).__init__(
81796$
        mean_squared_logarithmic_error, name=name, reduction=reduction)


@keras_export('keras.losses.BinaryCrossentropy')
class BinaryCrossentropy(LossFunctionWrapper):
81796$
  """Computes the cross-entropy loss between true labels and predicted labels.

81796$
  Use this cross-entropy loss for binary (0 or 1) classification applications.
81796$
  The loss function requires the following inputs:

81796$
  - `y_true` (true label): This is either 0 or 1.
81796$
  - `y_pred` (predicted value): This is the model's prediction, i.e, a single
81796$
    floating-point value which either represents a
81796$
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
81796$
    when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
81796$
    `from_logits=False`).

81796$
  **Recommended Usage:** (set `from_logits=True`)

81796$
  With `tf.keras` API:

81796$
  ```python
81796$
  model.compile(
81796$
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
81796$
    ....
81796$
  )
81796$
  ```

81796$
  As a standalone function:

81796$
  >>> # Example 1: (batch_size = 1, number of samples = 4)
81796$
  >>> y_true = [0, 1, 0, 0]
81796$
  >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
81796$
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
81796$
  >>> bce(y_true, y_pred).numpy()
81796$
  0.865

81796$
  >>> # Example 2: (batch_size = 2, number of samples = 4)
81796$
  >>> y_true = [[0, 1], [0, 0]]
81796$
  >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
81796$
  >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
81796$
  >>> bce(y_true, y_pred).numpy()
81796$
  0.865
81796$
  >>> # Using 'sample_weight' attribute
81796$
  >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
81796$
  0.243
81796$
  >>> # Using 'sum' reduction` type.
81796$
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> bce(y_true, y_pred).numpy()
81796$
  1.730
81796$
  >>> # Using 'none' reduction type.
81796$
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> bce(y_true, y_pred).numpy()
81796$
  array([0.235, 1.496], dtype=float32)

81796$
  **Default Usage:** (set `from_logits=False`)

81796$
  >>> # Make the following updates to the above "Recommended Usage" section
81796$
  >>> # 1. Set `from_logits=False`
81796$
  >>> tf.keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
81796$
  >>> # 2. Update `y_pred` to use probabilities instead of logits
81796$
  >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
81796$
  """

81796$
  def __init__(self,
81796$
               from_logits=False,
81796$
               label_smoothing=0,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='binary_crossentropy'):
81796$
    """Initializes `BinaryCrossentropy` instance.

81796$
    Args:
81796$
      from_logits: Whether to interpret `y_pred` as a tensor of
81796$
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
81796$
          assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
81796$
      label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
81796$
        we compute the loss between the predicted labels and a smoothed version
81796$
        of the true labels, where the smoothing squeezes the labels towards 0.5.
81796$
        Larger values of `label_smoothing` correspond to heavier smoothing.
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: (Optional) Name for the op. Defaults to 'binary_crossentropy'.
81796$
    """
81796$
    super(BinaryCrossentropy, self).__init__(
81796$
        binary_crossentropy,
81796$
        name=name,
81796$
        reduction=reduction,
81796$
        from_logits=from_logits,
81796$
        label_smoothing=label_smoothing)
81796$
    self.from_logits = from_logits


@keras_export('keras.losses.CategoricalCrossentropy')
class CategoricalCrossentropy(LossFunctionWrapper):
81796$
  """Computes the crossentropy loss between the labels and predictions.

81796$
  Use this crossentropy loss function when there are two or more label classes.
81796$
  We expect labels to be provided in a `one_hot` representation. If you want to
81796$
  provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
81796$
  There should be `# classes` floating point values per feature.

81796$
  In the snippet below, there is `# classes` floating pointing values per
81796$
  example. The shape of both `y_pred` and `y_true` are
81796$
  `[batch_size, num_classes]`.

81796$
  Standalone usage:

81796$
  >>> y_true = [[0, 1, 0], [0, 0, 1]]
81796$
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> cce = tf.keras.losses.CategoricalCrossentropy()
81796$
  >>> cce(y_true, y_pred).numpy()
81796$
  1.177

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
81796$
  0.814

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> cce = tf.keras.losses.CategoricalCrossentropy(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> cce(y_true, y_pred).numpy()
81796$
  2.354

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> cce = tf.keras.losses.CategoricalCrossentropy(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> cce(y_true, y_pred).numpy()
81796$
  array([0.0513, 2.303], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               from_logits=False,
81796$
               label_smoothing=0,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='categorical_crossentropy'):
81796$
    """Initializes `CategoricalCrossentropy` instance.

81796$
    Args:
81796$
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
81796$
        default, we assume that `y_pred` encodes a probability distribution.
81796$
      label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
81796$
        meaning the confidence on label values are relaxed. For example, if
81796$
        `0.1`, use `0.1 / num_classes` for non-target labels and
81796$
        `0.9 + 0.1 / num_classes` for target labels.
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'categorical_crossentropy'.
81796$
    """
81796$
    super(CategoricalCrossentropy, self).__init__(
81796$
        categorical_crossentropy,
81796$
        name=name,
81796$
        reduction=reduction,
81796$
        from_logits=from_logits,
81796$
        label_smoothing=label_smoothing)


@keras_export('keras.losses.SparseCategoricalCrossentropy')
class SparseCategoricalCrossentropy(LossFunctionWrapper):
81796$
  """Computes the crossentropy loss between the labels and predictions.

81796$
  Use this crossentropy loss function when there are two or more label classes.
81796$
  We expect labels to be provided as integers. If you want to provide labels
81796$
  using `one-hot` representation, please use `CategoricalCrossentropy` loss.
81796$
  There should be `# classes` floating point values per feature for `y_pred`
81796$
  and a single floating point value per feature for `y_true`.

81796$
  In the snippet below, there is a single floating point value per example for
81796$
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
81796$
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
81796$
  `[batch_size, num_classes]`.

81796$
  Standalone usage:

81796$
  >>> y_true = [1, 2]
81796$
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
81796$
  >>> scce(y_true, y_pred).numpy()
81796$
  1.177

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> scce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
81796$
  0.814

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> scce(y_true, y_pred).numpy()
81796$
  2.354

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> scce(y_true, y_pred).numpy()
81796$
  array([0.0513, 2.303], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd',
81796$
                loss=tf.keras.losses.SparseCategoricalCrossentropy())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               from_logits=False,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='sparse_categorical_crossentropy'):
81796$
    """Initializes `SparseCategoricalCrossentropy` instance.

81796$
    Args:
81796$
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
81796$
        default, we assume that `y_pred` encodes a probability distribution.
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to
81796$
        'sparse_categorical_crossentropy'.
81796$
    """
81796$
    super(SparseCategoricalCrossentropy, self).__init__(
81796$
        sparse_categorical_crossentropy,
81796$
        name=name,
81796$
        reduction=reduction,
81796$
        from_logits=from_logits)


@keras_export('keras.losses.Hinge')
class Hinge(LossFunctionWrapper):
81796$
  """Computes the hinge loss between `y_true` and `y_pred`.

81796$
  `loss = maximum(1 - y_true * y_pred, 0)`

81796$
  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
81796$
  provided we will convert them to -1 or 1.

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> h = tf.keras.losses.Hinge()
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  1.3

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
81796$
  0.55

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> h = tf.keras.losses.Hinge(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  2.6

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> h = tf.keras.losses.Hinge(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  array([1.1, 1.5], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.Hinge())
81796$
  ```
81796$
  """

81796$
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='hinge'):
81796$
    """Initializes `Hinge` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'hinge'.
81796$
    """
81796$
    super(Hinge, self).__init__(hinge, name=name, reduction=reduction)


@keras_export('keras.losses.SquaredHinge')
class SquaredHinge(LossFunctionWrapper):
81796$
  """Computes the squared hinge loss between `y_true` and `y_pred`.

81796$
  `loss = square(maximum(1 - y_true * y_pred, 0))`

81796$
  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
81796$
  provided we will convert them to -1 or 1.

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> h = tf.keras.losses.SquaredHinge()
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  1.86

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
81796$
  0.73

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> h = tf.keras.losses.SquaredHinge(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  3.72

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> h = tf.keras.losses.SquaredHinge(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  array([1.46, 2.26], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.SquaredHinge())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='squared_hinge'):
81796$
    """Initializes `SquaredHinge` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'squared_hinge'.
81796$
    """
81796$
    super(SquaredHinge, self).__init__(
81796$
        squared_hinge, name=name, reduction=reduction)


@keras_export('keras.losses.CategoricalHinge')
class CategoricalHinge(LossFunctionWrapper):
81796$
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

81796$
  `loss = maximum(neg - pos + 1, 0)`
81796$
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0, 1], [0, 0]]
81796$
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> h = tf.keras.losses.CategoricalHinge()
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  1.4

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
81796$
  0.6

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> h = tf.keras.losses.CategoricalHinge(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  2.8

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> h = tf.keras.losses.CategoricalHinge(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  array([1.2, 1.6], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalHinge())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='categorical_hinge'):
81796$
    """Initializes `CategoricalHinge` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'categorical_hinge'.
81796$
    """
81796$
    super(CategoricalHinge, self).__init__(
81796$
        categorical_hinge, name=name, reduction=reduction)


@keras_export('keras.losses.Poisson')
class Poisson(LossFunctionWrapper):
81796$
  """Computes the Poisson loss between `y_true` and `y_pred`.

81796$
  `loss = y_pred - y_true * log(y_pred)`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[1., 1.], [0., 0.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> p = tf.keras.losses.Poisson()
81796$
  >>> p(y_true, y_pred).numpy()
81796$
  0.5

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> p(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
81796$
  0.4

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> p = tf.keras.losses.Poisson(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> p(y_true, y_pred).numpy()
81796$
  0.999

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> p = tf.keras.losses.Poisson(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> p(y_true, y_pred).numpy()
81796$
  array([0.999, 0.], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.Poisson())
81796$
  ```
81796$
  """

81796$
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='poisson'):
81796$
    """Initializes `Poisson` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'poisson'.
81796$
    """
81796$
    super(Poisson, self).__init__(poisson, name=name, reduction=reduction)


@keras_export('keras.losses.LogCosh')
class LogCosh(LossFunctionWrapper):
81796$
  """Computes the logarithm of the hyperbolic cosine of the prediction error.

81796$
  `logcosh = log((exp(x) + exp(-x))/2)`,
81796$
  where x is the error `y_pred - y_true`.

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [0., 0.]]
81796$
  >>> y_pred = [[1., 1.], [0., 0.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> l = tf.keras.losses.LogCosh()
81796$
  >>> l(y_true, y_pred).numpy()
81796$
  0.108

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> l(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
81796$
  0.087

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> l = tf.keras.losses.LogCosh(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> l(y_true, y_pred).numpy()
81796$
  0.217

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> l = tf.keras.losses.LogCosh(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> l(y_true, y_pred).numpy()
81796$
  array([0.217, 0.], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.LogCosh())
81796$
  ```
81796$
  """

81796$
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'):
81796$
    """Initializes `LogCosh` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'log_cosh'.
81796$
    """
81796$
    super(LogCosh, self).__init__(log_cosh, name=name, reduction=reduction)


@keras_export('keras.losses.KLDivergence')
class KLDivergence(LossFunctionWrapper):
81796$
  """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

81796$
  `loss = y_true * log(y_true / y_pred)`

81796$
  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

81796$
  Standalone usage:

81796$
  >>> y_true = [[0, 1], [0, 0]]
81796$
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> kl = tf.keras.losses.KLDivergence()
81796$
  >>> kl(y_true, y_pred).numpy()
81796$
  0.458

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> kl(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
81796$
  0.366

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> kl = tf.keras.losses.KLDivergence(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> kl(y_true, y_pred).numpy()
81796$
  0.916

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> kl = tf.keras.losses.KLDivergence(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> kl(y_true, y_pred).numpy()
81796$
  array([0.916, -3.08e-06], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.KLDivergence())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='kl_divergence'):
81796$
    """Initializes `KLDivergence` instance.

81796$
    Args:
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'kl_divergence'.
81796$
    """
81796$
    super(KLDivergence, self).__init__(
81796$
        kl_divergence, name=name, reduction=reduction)


@keras_export('keras.losses.Huber')
class Huber(LossFunctionWrapper):
81796$
  """Computes the Huber loss between `y_true` and `y_pred`.

81796$
  For each value x in `error = y_true - y_pred`:

81796$
  ```
81796$
  loss = 0.5 * x^2                  if |x| <= d
81796$
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
81796$
  ```
81796$
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

81796$
  Standalone usage:

81796$
  >>> y_true = [[0, 1], [0, 0]]
81796$
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> h = tf.keras.losses.Huber()
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  0.155

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
81796$
  0.09

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> h = tf.keras.losses.Huber(
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  0.31

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> h = tf.keras.losses.Huber(
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> h(y_true, y_pred).numpy()
81796$
  array([0.18, 0.13], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.Huber())
81796$
  ```
81796$
  """

81796$
  def __init__(self,
81796$
               delta=1.0,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='huber_loss'):
81796$
    """Initializes `Huber` instance.

81796$
    Args:
81796$
      delta: A float, the point where the Huber loss function changes from a
81796$
        quadratic to linear.
81796$
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
81796$
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
81796$
        option will be determined by the usage context. For almost all cases
81796$
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
81796$
        `tf.distribute.Strategy`, outside of built-in training loops such as
81796$
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
81796$
        will raise an error. Please see this custom training [tutorial](
81796$
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
81796$
            more details.
81796$
      name: Optional name for the op. Defaults to 'huber_loss'.
81796$
    """
81796$
    super(Huber, self).__init__(
81796$
        huber, name=name, reduction=reduction, delta=delta)


@keras_export('keras.metrics.mean_squared_error', 'keras.metrics.mse',
81796$
              'keras.metrics.MSE', 'keras.losses.mean_squared_error',
81796$
              'keras.losses.mse', 'keras.losses.MSE')
@dispatch.add_dispatch_support
def mean_squared_error(y_true, y_pred):
81796$
  """Computes the mean squared error between labels and predictions.

81796$
  After computing the squared distance between the inputs, the mean value over
81796$
  the last dimension is returned.

81796$
  `loss = mean(square(y_true - y_pred), axis=-1)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.randint(0, 2, size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> assert np.array_equal(
81796$
  ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  return backend.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)


def _ragged_tensor_apply_loss(loss_fn, y_true, y_pred):
81796$
  """Apply a loss function on a per batch basis.

81796$
  Args:
81796$
    loss_fn: The loss function
81796$
    y_true: truth values (RaggedTensor)
81796$
    y_pred: predicted values (RaggedTensor)

81796$
  Returns:
81796$
    Loss-function result. A dense tensor if the output has a single dimension
81796$
    (per-batch loss value); a ragged tensor otherwise.
81796$
  """

81796$
  def rt_is_equiv_dense(rt):
81796$
    """Returns true if this RaggedTensor has the same row_lenghts across

81796$
       all ragged dimensions and thus can be converted to a dense tensor
81796$
       without loss of information.

81796$
    Args:
81796$
      rt: RaggedTensor.
81796$
    """
81796$
    return math_ops.reduce_all([
81796$
        math_ops.equal(
81796$
            math_ops.reduce_variance(math_ops.cast(row_lens, backend.floatx())),
81796$
            constant_op.constant([0.])) for row_lens in rt.nested_row_lengths()
81796$
    ])

81796$
  def _convert_to_dense(inputs):
81796$
    return tuple(rt.to_tensor() for rt in inputs)

81796$
  def _wrapper(inputs):
81796$
    _, y_pred = inputs
81796$
    if isinstance(y_pred, ragged_tensor.RaggedTensor):
81796$
      return control_flow_ops.cond(
81796$
          rt_is_equiv_dense(y_pred),
81796$
          lambda: loss_fn(*_convert_to_dense(inputs)), lambda: loss_fn(*inputs))

81796$
    return loss_fn(*inputs)

81796$
  lshape = y_pred.shape.as_list()[1:-1]
81796$
  if len(lshape) > 0:
81796$
    spec = ragged_tensor.RaggedTensorSpec(shape=lshape, dtype=y_pred.dtype)
81796$
  else:
81796$
    spec = tensor_spec.TensorSpec(shape=[], dtype=y_pred.dtype)

81796$
  nested_splits_list = [rt.nested_row_splits for rt in (y_true, y_pred)]
81796$
  assertion_list = ragged_util.assert_splits_match(nested_splits_list)
81796$
  with ops.control_dependencies(assertion_list):
81796$
    return ragged_map_ops.map_fn(_wrapper, elems=(y_true, y_pred), dtype=spec)


@dispatch.dispatch_for_types(mean_squared_error, ragged_tensor.RaggedTensor)
def _ragged_tensor_mse(y_true, y_pred):
81796$
  """Implements support for handling RaggedTensors.

81796$
  Args:
81796$
    y_true: RaggedTensor truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: RaggedTensor predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
81796$
    When the number of dimensions of the batch feature vector [d0, .. dN] is
81796$
    greater than one the return value is a RaggedTensor. Otherwise a Dense
81796$
    tensor with dimensions [batch_size] is returned.
81796$
  """
81796$
  return _ragged_tensor_apply_loss(mean_squared_error, y_true, y_pred)


@keras_export('keras.metrics.mean_absolute_error', 'keras.metrics.mae',
81796$
              'keras.metrics.MAE', 'keras.losses.mean_absolute_error',
81796$
              'keras.losses.mae', 'keras.losses.MAE')
@dispatch.add_dispatch_support
def mean_absolute_error(y_true, y_pred):
81796$
  """Computes the mean absolute error between labels and predictions.

81796$
  `loss = mean(abs(y_true - y_pred), axis=-1)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.randint(0, 2, size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> assert np.array_equal(
81796$
  ...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  return backend.mean(math_ops.abs(y_pred - y_true), axis=-1)


@dispatch.dispatch_for_types(mean_absolute_error, ragged_tensor.RaggedTensor)
def _ragged_tensor_mae(y_true, y_pred):
81796$
  """RaggedTensor adapter for mean_absolute_error."""
81796$
  return _ragged_tensor_apply_loss(mean_absolute_error, y_true, y_pred)


@keras_export('keras.metrics.mean_absolute_percentage_error',
81796$
              'keras.metrics.mape', 'keras.metrics.MAPE',
81796$
              'keras.losses.mean_absolute_percentage_error',
81796$
              'keras.losses.mape', 'keras.losses.MAPE')
@dispatch.add_dispatch_support
def mean_absolute_percentage_error(y_true, y_pred):
81796$
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

81796$
  `loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.random(size=(2, 3))
81796$
  >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> assert np.array_equal(
81796$
  ...     loss.numpy(),
81796$
  ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  diff = math_ops.abs(
81796$
      (y_true - y_pred) / backend.maximum(math_ops.abs(y_true),
81796$
                                          backend.epsilon()))
81796$
  return 100. * backend.mean(diff, axis=-1)


@dispatch.dispatch_for_types(mean_absolute_percentage_error,
81796$
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_mape(y_true, y_pred):
81796$
  """Support RaggedTensors."""
81796$
  return _ragged_tensor_apply_loss(mean_absolute_percentage_error, y_true,
81796$
                                   y_pred)


@keras_export('keras.metrics.mean_squared_logarithmic_error',
81796$
              'keras.metrics.msle', 'keras.metrics.MSLE',
81796$
              'keras.losses.mean_squared_logarithmic_error',
81796$
              'keras.losses.msle', 'keras.losses.MSLE')
@dispatch.add_dispatch_support
def mean_squared_logarithmic_error(y_true, y_pred):
81796$
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

81796$
  `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.randint(0, 2, size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> y_true = np.maximum(y_true, 1e-7)
81796$
  >>> y_pred = np.maximum(y_pred, 1e-7)
81796$
  >>> assert np.allclose(
81796$
  ...     loss.numpy(),
81796$
  ...     np.mean(
81796$
  ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  first_log = math_ops.log(backend.maximum(y_pred, backend.epsilon()) + 1.)
81796$
  second_log = math_ops.log(backend.maximum(y_true, backend.epsilon()) + 1.)
81796$
  return backend.mean(
81796$
      math_ops.squared_difference(first_log, second_log), axis=-1)


@dispatch.dispatch_for_types(mean_squared_logarithmic_error,
81796$
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_msle(y_true, y_pred):
81796$
  """Implements support for handling RaggedTensors."""
81796$
  return _ragged_tensor_apply_loss(mean_squared_logarithmic_error, y_true,
81796$
                                   y_pred)


def _maybe_convert_labels(y_true):
81796$
  """Converts binary labels into -1/1."""
81796$
  are_zeros = math_ops.equal(y_true, 0)
81796$
  are_ones = math_ops.equal(y_true, 1)
81796$
  is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

81796$
  def _convert_binary_labels():
81796$
    # Convert the binary labels to -1 or 1.
81796$
    return 2. * y_true - 1.

81796$
  updated_y_true = smart_cond.smart_cond(is_binary, _convert_binary_labels,
81796$
                                         lambda: y_true)
81796$
  return updated_y_true


@keras_export('keras.metrics.squared_hinge', 'keras.losses.squared_hinge')
@dispatch.add_dispatch_support
def squared_hinge(y_true, y_pred):
81796$
  """Computes the squared hinge loss between `y_true` and `y_pred`.

81796$
  `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> assert np.array_equal(
81796$
  ...     loss.numpy(),
81796$
  ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))

81796$
  Args:
81796$
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
81796$
      If binary (0 or 1) labels are provided we will convert them to -1 or 1.
81796$
      shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
     Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  y_true = _maybe_convert_labels(y_true)
81796$
  return backend.mean(
81796$
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)


@keras_export('keras.metrics.hinge', 'keras.losses.hinge')
@dispatch.add_dispatch_support
def hinge(y_true, y_pred):
81796$
  """Computes the hinge loss between `y_true` and `y_pred`.

81796$
  `loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.hinge(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> assert np.array_equal(
81796$
  ...     loss.numpy(),
81796$
  ...     np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1))

81796$
  Args:
81796$
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
81796$
      If binary (0 or 1) labels are provided they will be converted to -1 or 1.
81796$
      shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  y_true = _maybe_convert_labels(y_true)
81796$
  return backend.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)


@keras_export('keras.losses.categorical_hinge')
@dispatch.add_dispatch_support
def categorical_hinge(y_true, y_pred):
81796$
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

81796$
  `loss = maximum(neg - pos + 1, 0)`
81796$
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.randint(0, 3, size=(2,))
81796$
  >>> y_true = tf.keras.utils.to_categorical(y_true, num_classes=3)
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> pos = np.sum(y_true * y_pred, axis=-1)
81796$
  >>> neg = np.amax((1. - y_true) * y_pred, axis=-1)
81796$
  >>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))

81796$
  Args:
81796$
    y_true: The ground truth values. `y_true` values are expected to be
81796$
    either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor).
81796$
    y_pred: The predicted values.

81796$
  Returns:
81796$
    Categorical hinge loss values.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
81796$
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
81796$
  zero = math_ops.cast(0., y_pred.dtype)
81796$
  return math_ops.maximum(neg - pos + 1., zero)


@keras_export('keras.losses.huber', v1=[])
@dispatch.add_dispatch_support
def huber(y_true, y_pred, delta=1.0):
81796$
  """Computes Huber loss value.

81796$
  For each value x in `error = y_true - y_pred`:

81796$
  ```
81796$
  loss = 0.5 * x^2                  if |x| <= d
81796$
  loss = d * |x| - 0.5 * d^2        if |x| > d
81796$
  ```
81796$
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

81796$
  Args:
81796$
    y_true: tensor of true targets.
81796$
    y_pred: tensor of predicted targets.
81796$
    delta: A float, the point where the Huber loss function changes from a
81796$
      quadratic to linear.

81796$
  Returns:
81796$
    Tensor with one scalar loss entry per sample.
81796$
  """
81796$
  y_pred = math_ops.cast(y_pred, dtype=backend.floatx())
81796$
  y_true = math_ops.cast(y_true, dtype=backend.floatx())
81796$
  delta = math_ops.cast(delta, dtype=backend.floatx())
81796$
  error = math_ops.subtract(y_pred, y_true)
81796$
  abs_error = math_ops.abs(error)
81796$
  half = ops.convert_to_tensor_v2_with_dispatch(0.5, dtype=abs_error.dtype)
81796$
  return backend.mean(
81796$
      array_ops.where_v2(abs_error <= delta, half * math_ops.square(error),
81796$
                         delta * abs_error - half * math_ops.square(delta)),
81796$
      axis=-1)


@keras_export('keras.losses.log_cosh', 'keras.losses.logcosh',
81796$
              'keras.metrics.log_cosh', 'keras.metrics.logcosh')
@dispatch.add_dispatch_support
def log_cosh(y_true, y_pred):
81796$
  """Logarithm of the hyperbolic cosine of the prediction error.

81796$
  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
81796$
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
81796$
  like the mean squared error, but will not be so strongly affected by the
81796$
  occasional wildly incorrect prediction.

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.random(size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.logcosh(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> x = y_pred - y_true
81796$
  >>> assert np.allclose(
81796$
  ...     loss.numpy(),
81796$
  ...     np.mean(x + np.log(np.exp(-2. * x) + 1.) - math_ops.log(2.), axis=-1),
81796$
  ...     atol=1e-5)

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
    Logcosh error values. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)

81796$
  def _logcosh(x):
81796$
    return x + math_ops.softplus(-2. * x) - math_ops.cast(
81796$
        math_ops.log(2.), x.dtype)

81796$
  return backend.mean(_logcosh(y_pred - y_true), axis=-1)


@keras_export('keras.metrics.categorical_crossentropy',
81796$
              'keras.losses.categorical_crossentropy')
@dispatch.add_dispatch_support
def categorical_crossentropy(y_true,
81796$
                             y_pred,
81796$
                             from_logits=False,
81796$
                             label_smoothing=0,
81796$
                             axis=-1):
81796$
  """Computes the categorical crossentropy loss.

81796$
  Standalone usage:

81796$
  >>> y_true = [[0, 1, 0], [0, 0, 1]]
81796$
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
81796$
  >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> loss.numpy()
81796$
  array([0.0513, 2.303], dtype=float32)

81796$
  Args:
81796$
    y_true: Tensor of one-hot true targets.
81796$
    y_pred: Tensor of predicted targets.
81796$
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
81796$
      we assume that `y_pred` encodes a probability distribution.
81796$
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
81796$
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
81796$
      and `0.9 + 0.1 / num_classes` for target labels.
81796$
    axis: (Optional) Defaults to -1. The dimension along which the entropy is
81796$
      computed.

81796$
  Returns:
81796$
    Categorical crossentropy loss value.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  label_smoothing = ops.convert_to_tensor_v2_with_dispatch(
81796$
      label_smoothing, dtype=backend.floatx())

81796$
  def _smooth_labels():
81796$
    num_classes = math_ops.cast(array_ops.shape(y_true)[-1], y_pred.dtype)
81796$
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

81796$
  y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels,
81796$
                                 lambda: y_true)
81796$
  return backend.categorical_crossentropy(
81796$
      y_true, y_pred, from_logits=from_logits, axis=axis)


@dispatch.dispatch_for_types(categorical_crossentropy,
81796$
                             ragged_tensor.RaggedTensor)
def _ragged_tensor_categorical_crossentropy(y_true,
81796$
                                            y_pred,
81796$
                                            from_logits=False,
81796$
                                            label_smoothing=0):
81796$
  """Implements support for handling RaggedTensors.

81796$
  Args:
81796$
    y_true: Tensor of one-hot true targets.
81796$
    y_pred: Tensor of predicted targets.
81796$
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
81796$
      we assume that `y_pred` encodes a probability distribution.
81796$
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
81796$
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
81796$
      and `0.9 + 0.1 / num_classes` for target labels.

81796$
  Returns:
81796$
    Categorical crossentropy loss value.

81796$
  Expected shape: (batch, sequence_len, n_classes) with sequence_len
81796$
  being variable per batch.
81796$
  Return shape: (batch, sequence_len).

81796$
  When used by CategoricalCrossentropy() with the default reduction
81796$
  (SUM_OVER_BATCH_SIZE), the reduction averages the loss over the
81796$
  number of elements independent of the batch. E.g. if the RaggedTensor
81796$
  has 2 batches with [2, 1] values respectivly the resulting loss is
81796$
  the sum of the individual loss values divided by 3.
81796$
  """
81796$
  fn = functools.partial(
81796$
      categorical_crossentropy,
81796$
      from_logits=from_logits,
81796$
      label_smoothing=label_smoothing)
81796$
  return _ragged_tensor_apply_loss(fn, y_true, y_pred)


@keras_export('keras.metrics.sparse_categorical_crossentropy',
81796$
              'keras.losses.sparse_categorical_crossentropy')
@dispatch.add_dispatch_support
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
81796$
  """Computes the sparse categorical crossentropy loss.

81796$
  Standalone usage:

81796$
  >>> y_true = [1, 2]
81796$
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
81796$
  >>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> loss.numpy()
81796$
  array([0.0513, 2.303], dtype=float32)

81796$
  Args:
81796$
    y_true: Ground truth values.
81796$
    y_pred: The predicted values.
81796$
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
81796$
      we assume that `y_pred` encodes a probability distribution.
81796$
    axis: (Optional) Defaults to -1. The dimension along which the entropy is
81796$
      computed.

81796$
  Returns:
81796$
    Sparse categorical crossentropy loss value.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  return backend.sparse_categorical_crossentropy(
81796$
      y_true, y_pred, from_logits=from_logits, axis=axis)


@keras_export('keras.metrics.binary_crossentropy',
81796$
              'keras.losses.binary_crossentropy')
@dispatch.add_dispatch_support
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0, axis=-1):
81796$
  """Computes the binary crossentropy loss.

81796$
  Standalone usage:

81796$
  >>> y_true = [[0, 1], [0, 0]]
81796$
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
81796$
  >>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> loss.numpy()
81796$
  array([0.916 , 0.714], dtype=float32)

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
81796$
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
81796$
      we assume that `y_pred` encodes a probability distribution.
81796$
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels by
81796$
      squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
81796$
      for the target class and `0.5 * label_smoothing` for the non-target class.
81796$
    axis: (Optional) Defaults to -1. The dimension along which the mean is
81796$
      computed.

81796$
  Returns:
81796$
    Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  label_smoothing = ops.convert_to_tensor_v2_with_dispatch(
81796$
      label_smoothing, dtype=backend.floatx())

81796$
  def _smooth_labels():
81796$
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

81796$
  y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels,
81796$
                                 lambda: y_true)
81796$
  return backend.mean(
81796$
      backend.binary_crossentropy(
81796$
          y_true, y_pred, from_logits=from_logits), axis=axis)


@dispatch.dispatch_for_types(binary_crossentropy, ragged_tensor.RaggedTensor)
def _ragged_tensor_binary_crossentropy(y_true,
81796$
                                       y_pred,
81796$
                                       from_logits=False,
81796$
                                       label_smoothing=0):
81796$
  """Implements support for handling RaggedTensors.

81796$
  Args:
81796$
    y_true: Tensor of one-hot true targets.
81796$
    y_pred: Tensor of predicted targets.
81796$
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
81796$
      we assume that `y_pred` encodes a probability distribution.
81796$
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
81796$
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
81796$
      and `0.9 + 0.1 / num_classes` for target labels.

81796$
  Returns:
81796$
    Binary crossentropy loss value.

81796$
  Expected shape: (batch, sequence_len) with sequence_len being variable
81796$
  per batch.
81796$
  Return shape: (batch,); returns the per batch mean of the loss values.

81796$
  When used by BinaryCrossentropy() with the default reduction
81796$
  (SUM_OVER_BATCH_SIZE), the reduction averages the per batch losses over
81796$
  the number of batches.
81796$
  """
81796$
  fn = functools.partial(
81796$
      binary_crossentropy,
81796$
      from_logits=from_logits,
81796$
      label_smoothing=label_smoothing)
81796$
  return _ragged_tensor_apply_loss(fn, y_true, y_pred)


@keras_export('keras.metrics.kl_divergence',
81796$
              'keras.metrics.kullback_leibler_divergence', 'keras.metrics.kld',
81796$
              'keras.metrics.KLD', 'keras.losses.kl_divergence',
81796$
              'keras.losses.kullback_leibler_divergence', 'keras.losses.kld',
81796$
              'keras.losses.KLD')
@dispatch.add_dispatch_support
def kl_divergence(y_true, y_pred):
81796$
  """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

81796$
  `loss = y_true * log(y_true / y_pred)`

81796$
  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
81796$
  >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
81796$
  >>> assert np.array_equal(
81796$
  ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))

81796$
  Args:
81796$
    y_true: Tensor of true targets.
81796$
    y_pred: Tensor of predicted targets.

81796$
  Returns:
81796$
    A `Tensor` with loss.

81796$
  Raises:
81796$
    TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  y_true = backend.clip(y_true, backend.epsilon(), 1)
81796$
  y_pred = backend.clip(y_pred, backend.epsilon(), 1)
81796$
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)


@keras_export('keras.metrics.poisson', 'keras.losses.poisson')
@dispatch.add_dispatch_support
def poisson(y_true, y_pred):
81796$
  """Computes the Poisson loss between y_true and y_pred.

81796$
  The Poisson loss is the mean of the elements of the `Tensor`
81796$
  `y_pred - y_true * log(y_pred)`.

81796$
  Standalone usage:

81796$
  >>> y_true = np.random.randint(0, 2, size=(2, 3))
81796$
  >>> y_pred = np.random.random(size=(2, 3))
81796$
  >>> loss = tf.keras.losses.poisson(y_true, y_pred)
81796$
  >>> assert loss.shape == (2,)
81796$
  >>> y_pred = y_pred + 1e-7
81796$
  >>> assert np.allclose(
81796$
  ...     loss.numpy(), np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
81796$
  ...     atol=1e-5)

81796$
  Args:
81796$
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
81796$
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

81796$
  Returns:
81796$
     Poisson loss value. shape = `[batch_size, d0, .. dN-1]`.

81796$
  Raises:
81796$
    InvalidArgumentError: If `y_true` and `y_pred` have incompatible shapes.
81796$
  """
81796$
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
81796$
  y_true = math_ops.cast(y_true, y_pred.dtype)
81796$
  return backend.mean(
81796$
      y_pred - y_true * math_ops.log(y_pred + backend.epsilon()), axis=-1)


@keras_export(
81796$
    'keras.losses.cosine_similarity',
81796$
    v1=[
81796$
        'keras.metrics.cosine_proximity',
81796$
        'keras.metrics.cosine',
81796$
        'keras.losses.cosine_proximity',
81796$
        'keras.losses.cosine',
81796$
        'keras.losses.cosine_similarity',
81796$
    ])
@dispatch.add_dispatch_support
def cosine_similarity(y_true, y_pred, axis=-1):
81796$
  """Computes the cosine similarity between labels and predictions.

81796$
  Note that it is a number between -1 and 1. When it is a negative number
81796$
  between -1 and 0, 0 indicates orthogonality and values closer to -1
81796$
  indicate greater similarity. The values closer to 1 indicate greater
81796$
  dissimilarity. This makes it usable as a loss function in a setting
81796$
  where you try to maximize the proximity between predictions and
81796$
  targets. If either `y_true` or `y_pred` is a zero vector, cosine
81796$
  similarity will be 0 regardless of the proximity between predictions
81796$
  and targets.

81796$
  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
81796$
  >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
81796$
  >>> loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
81796$
  >>> loss.numpy()
81796$
  array([-0., -0.999, 0.999], dtype=float32)

81796$
  Args:
81796$
    y_true: Tensor of true targets.
81796$
    y_pred: Tensor of predicted targets.
81796$
    axis: Axis along which to determine similarity.

81796$
  Returns:
81796$
    Cosine similarity tensor.
81796$
  """
81796$
  y_true = nn.l2_normalize(y_true, axis=axis)
81796$
  y_pred = nn.l2_normalize(y_pred, axis=axis)
81796$
  return -math_ops.reduce_sum(y_true * y_pred, axis=axis)


@keras_export('keras.losses.CosineSimilarity')
class CosineSimilarity(LossFunctionWrapper):
81796$
  """Computes the cosine similarity between labels and predictions.

81796$
  Note that it is a number between -1 and 1. When it is a negative number
81796$
  between -1 and 0, 0 indicates orthogonality and values closer to -1
81796$
  indicate greater similarity. The values closer to 1 indicate greater
81796$
  dissimilarity. This makes it usable as a loss function in a setting
81796$
  where you try to maximize the proximity between predictions and targets.
81796$
  If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0
81796$
  regardless of the proximity between predictions and targets.

81796$
  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

81796$
  Standalone usage:

81796$
  >>> y_true = [[0., 1.], [1., 1.]]
81796$
  >>> y_pred = [[1., 0.], [1., 1.]]
81796$
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
81796$
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
81796$
  >>> # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
81796$
  >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
81796$
  >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
81796$
  >>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
81796$
  >>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
81796$
  >>> cosine_loss(y_true, y_pred).numpy()
81796$
  -0.5

81796$
  >>> # Calling with 'sample_weight'.
81796$
  >>> cosine_loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
81796$
  -0.0999

81796$
  >>> # Using 'sum' reduction type.
81796$
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
81796$
  ...     reduction=tf.keras.losses.Reduction.SUM)
81796$
  >>> cosine_loss(y_true, y_pred).numpy()
81796$
  -0.999

81796$
  >>> # Using 'none' reduction type.
81796$
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
81796$
  ...     reduction=tf.keras.losses.Reduction.NONE)
81796$
  >>> cosine_loss(y_true, y_pred).numpy()
81796$
  array([-0., -0.999], dtype=float32)

81796$
  Usage with the `compile()` API:

81796$
  ```python
81796$
  model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
81796$
  ```

81796$
  Args:
81796$
    axis: (Optional) Defaults to -1. The dimension along which the cosine
81796$
      similarity is computed.
81796$
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
81796$
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
81796$
      be determined by the usage context. For almost all cases this defaults to
81796$
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
81796$
      built-in training loops such as `tf.keras` `compile` and `fit`, using
81796$
      `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this
81796$
      custom training [tutorial]
81796$
      (https://www.tensorflow.org/tutorials/distribute/custom_training) for more
81796$
        details.
81796$
    name: Optional name for the op.
81796$
  """

81796$
  def __init__(self,
81796$
               axis=-1,
81796$
               reduction=losses_utils.ReductionV2.AUTO,
81796$
               name='cosine_similarity'):
81796$
    super(CosineSimilarity, self).__init__(
81796$
        cosine_similarity, reduction=reduction, name=name, axis=axis)


# Aliases.

bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
huber_loss = huber


def is_categorical_crossentropy(loss):
81796$
  result = ((isinstance(loss, CategoricalCrossentropy) or
81796$
             (isinstance(loss, LossFunctionWrapper) and
81796$
              loss.fn == categorical_crossentropy) or
81796$
             (hasattr(loss, '__name__') and
81796$
              loss.__name__ == 'categorical_crossentropy') or
81796$
             (loss == 'categorical_crossentropy')))
81796$
  return result


@keras_export('keras.losses.serialize')
def serialize(loss):
81796$
  """Serializes loss function or `Loss` instance.

81796$
  Args:
81796$
    loss: A Keras `Loss` instance or a loss function.

81796$
  Returns:
81796$
    Loss configuration dictionary.
81796$
  """
81796$
  return serialize_keras_object(loss)


@keras_export('keras.losses.deserialize')
def deserialize(name, custom_objects=None):
81796$
  """Deserializes a serialized loss class/function instance.

81796$
  Args:
81796$
      name: Loss configuration.
81796$
      custom_objects: Optional dictionary mapping names (strings) to custom
81796$
        objects (classes and functions) to be considered during deserialization.

81796$
  Returns:
81796$
      A Keras `Loss` instance or a loss function.
81796$
  """
81796$
  return deserialize_keras_object(
81796$
      name,
81796$
      module_objects=globals(),
81796$
      custom_objects=custom_objects,
81796$
      printable_module_name='loss function')


@keras_export('keras.losses.get')
def get(identifier):
81796$
  """Retrieves a Keras loss as a `function`/`Loss` class instance.

81796$
  The `identifier` may be the string name of a loss function or `Loss` class.

81796$
  >>> loss = tf.keras.losses.get("categorical_crossentropy")
81796$
  >>> type(loss)
81796$
  <class 'function'>
81796$
  >>> loss = tf.keras.losses.get("CategoricalCrossentropy")
81796$
  >>> type(loss)
81796$
  <class '...tensorflow.python.keras.losses.CategoricalCrossentropy'>

81796$
  You can also specify `config` of the loss to this function by passing dict
81796$
  containing `class_name` and `config` as an identifier. Also note that the
81796$
  `class_name` must map to a `Loss` class

81796$
  >>> identifier = {"class_name": "CategoricalCrossentropy",
81796$
  ...               "config": {"from_logits": True}}
81796$
  >>> loss = tf.keras.losses.get(identifier)
81796$
  >>> type(loss)
81796$
  <class '...tensorflow.python.keras.losses.CategoricalCrossentropy'>

81796$
  Args:
81796$
    identifier: A loss identifier. One of None or string name of a loss
81796$
      function/class or loss configuration dictionary or a loss function or a
81796$
      loss class instance.

81796$
  Returns:
81796$
    A Keras loss as a `function`/ `Loss` class instance.

81796$
  Raises:
81796$
    ValueError: If `identifier` cannot be interpreted.
81796$
  """
81796$
  if identifier is None:
81796$
    return None
81796$
  if isinstance(identifier, str):
81796$
    identifier = str(identifier)
81796$
    return deserialize(identifier)
81796$
  if isinstance(identifier, dict):
81796$
    return deserialize(identifier)
81796$
  elif callable(identifier):
81796$
    return identifier
81796$
  else:
81796$
    raise ValueError(
81796$
        'Could not interpret loss function identifier: {}'.format(identifier))


LABEL_DTYPES_FOR_LOSSES = {
81796$
    losses_impl.sparse_softmax_cross_entropy: 'int32',
81796$
    sparse_categorical_crossentropy: 'int32'
}
