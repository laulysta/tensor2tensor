# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import wsj_parsing
from tensor2tensor.utils import registry

import tensorflow as tf

from tensor2tensor.data_generators.wmt import *

FLAGS = tf.flags.FLAGS

import pdb
# End-of-sentence marker.
EOS = text_encoder.EOS_ID


class TranslateTemplate(problem.Text2TextProblem):
  """Base class for translation problems."""

  @property
  def is_character_level(self):
    return False

  @property
  def num_shards(self):
    return 100

  @property
  def vocab_name(self):
    return "vocab.pronoun-en"

  @property
  def use_subword_tokenizer(self):
    return True


@registry.register_problem
class TranslatePronouns(TranslateTemplate):
  """Problem spec for WMT En-De translation."""

  @property
  def targeted_vocab_size(self):
    return 20000

  def generator(self, data_dir, tmp_dir, train):
    #pdb.set_trace()

    source_vocab_size = self.targeted_vocab_size
    target_vocab_size = self.targeted_vocab_size

    #symbolizer_vocab = generator_utils.get_or_generate_vocab(data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size)


    source_datasets = [["pronoun_enfr_train.lang1",["pronoun_enfr_train.lang1"]]]
    target_datasets = [["pronoun_enfr_train.lang2",["pronoun_enfr_train.lang2"]]]


    source_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, "vocab.pronoun-en.%d" % source_vocab_size,
        source_vocab_size, source_datasets)
    target_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, "vocab.pronoun-fr.%d" % target_vocab_size,
        target_vocab_size, target_datasets)


    #datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    #tag = "train" if train else "dev"
    #data_path = _compile_data(tmp_dir, datasets, "wmt_ende_tok_%s" % tag)
    data_path = tmp_dir + "/pronoun_enfr_"
    if train:
      data_path = data_path + "train"
    else:
      data_path = data_path + "dev"

    return bi_vocabs_token_generator(data_path + ".lang2", data_path + ".lang1",
                                     source_vocab, target_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.FR_TOK