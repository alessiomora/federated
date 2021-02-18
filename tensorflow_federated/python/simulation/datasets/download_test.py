# Copyright 2021, The TensorFlow Federated Authors.
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
"""Tests for tensorflow_federated.python.simulation.datasets.download."""

import os
from unittest import mock

from absl import flags
from absl.testing import absltest
import lzma

from tensorflow_federated.python.simulation.datasets import download

FLAGS = flags.FLAGS


class DownloadTest(absltest.TestCase):

  @mock.patch('tensorflow.io.gfile.exists', return_value=False)
  def test_uncache_file_is_fetched_with_content_length(self, mock_gfile):
    test_data = b'data'
    test_url = 'http://www.test.org/my/test/file.lzma'
    mock_urlopen = mock.mock_open(read_data=lzma.compress(test_data))
    mock_urlopen.return_value.headers = {'context-length': str(len(test_data))}
    with mock.patch('urllib.request.urlopen', mock_urlopen):
      path = download.get_compressed_file(test_url, cache_dir=FLAGS.test_tmpdir)
    expected_output_path = os.path.join(FLAGS.test_tmpdir, 'file')
    self.assertEqual(path, expected_output_path)
    mock_gfile.assert_called_once_with(expected_output_path)
    mock_urlopen.assert_called_once_with(test_url)
    self.assertTrue(os.path.exists(expected_output_path))
    with open(expected_output_path, 'rb') as test_file:
      self.assertEqual(test_file.read(), test_data)

  @mock.patch('tensorflow.io.gfile.exists', return_value=False)
  def test_uncache_file_is_fetched_without_content_length(self, mock_gfile):
    test_data = b'data'
    test_url = 'http://www.test.org/my/test/file.lzma'
    mock_urlopen = mock.mock_open(read_data=lzma.compress(test_data))
    # Do not add content-length headers, ensuring that the python code
    # doesn't choke if the HTTP header was missing.
    mock_urlopen.return_value.headers = {}
    with mock.patch('urllib.request.urlopen', mock_urlopen):
      path = download.get_compressed_file(test_url, cache_dir=FLAGS.test_tmpdir)
    expected_output_path = os.path.join(FLAGS.test_tmpdir, 'file')
    self.assertEqual(path, expected_output_path)
    mock_gfile.assert_called_once_with(expected_output_path)
    mock_urlopen.assert_called_once_with(test_url)
    self.assertTrue(os.path.exists(expected_output_path))
    with open(expected_output_path, 'rb') as test_file:
      self.assertEqual(test_file.read(), test_data)

  @mock.patch('tensorflow.io.gfile.exists', return_value=True)
  def test_cached_file_is_not_fetched(self, mock_gfile):
    download.get_compressed_file('http://www.test.org/my/test/file.lzma')
    mock_gfile.assert_called_once()

  def test_non_lzma_extension_errors(self):
    with self.assertRaises(ValueError):
      download.get_compressed_file('file.bz2',
                                   'http://www.test.org/my/test/file.bz2')


if __name__ == '__main__':
  absltest.main()
