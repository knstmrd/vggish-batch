# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

r"""A simple demonstration of running VGGish in batch inference mode.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run WAV files through the model and write the embeddings to a .npy file. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_batch_inference.py --wav_files /path/to/a/text/file/containing/wav/file/paths

  # Run WAV files (as listed in a text file) through the model and also write the embeddings to
  # a .npy file (of shape (# wav files, 128)). The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_files /path/to/a/text/file/containing/wav/file/paths \
                                    --npy_file /path/to/output.npy \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params
  # In case a wav file is too short (less than 960ms), it is concatenated with itself until its length exceeds 960ms
  # If a wav file's length is 0, it is replaced with a 960-ms long array of zeros.
"""

from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    'wav_files', None,
    'Path to a file containing paths of wav files. Should contain signed 16-bit PCM samples.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'npy_file', 'output.npy',
    'Path to a .npy file where embeddings will be written.')

FLAGS = flags.FLAGS


def main(_):
  with open(FLAGS.wav_files) as f:
      files_list = [line.replace('\n', '') for line in f]

  n_files = len(files_list)
  output_emedding = np.zeros((n_files, 128))
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
  processed_fnames = []
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    for n_file, wav_file in enumerate(files_list):
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        print(n_file, '/', n_files)

        if examples_batch.shape[0] == 0:
          with open('bad_files.log', 'a') as logf:
            logf.write(wav_file + '\n')
        else:
          processed_fnames.append(wav_file)

          [embedding_batch] = sess.run([embedding_tensor],
                                       feed_dict={features_tensor: examples_batch})
          postprocessed_batch = pproc.postprocess(embedding_batch)
          postprocessed_batch_mean = np.mean(postprocessed_batch, axis=0)
          output_emedding[n_file, :] = postprocessed_batch_mean
      
    np.save(FLAGS.npy_file, output_emedding)


if __name__ == '__main__':
  tf.app.run()
