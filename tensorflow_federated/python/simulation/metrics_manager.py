# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility class for saving and loading simulation metrics."""

import abc
from typing import Mapping


class MetricsManager(metaclass=abc.ABCMeta):
  """An abstract base class for metrics managers.

  A `MetricsManager` is a utility to save metric data across a number of
  rounds of some simulation.
  """

  @abc.abstractmethod
  def save_metrics(self, round_num: int, metrics: Mapping[str, float]):
    """Updates the metrics manager with metrics for a given round.

    Args:
      round_num: A nonnegative integer representing the round number associated
        with `metrics`.
      metrics: A mapping from string values to floats.
    """
    raise NotImplementedError

  def clear_metrics(self, round_num: int):
    """Clear out metrics at or after a given starting `round_num`."""
    pass
