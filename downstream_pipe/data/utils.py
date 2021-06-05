# Copyright 2021 san kim
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

import gin
import seqio

from .vocabularies import HFSentencePieceVocabulary

DEFAULT_SPM_PATH = "gs://ket5/vocabs/ket5.64000/sentencepiece.model"
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)

@gin.configurable
def get_vocabulary(vocab_name=None):
    if vocab_name is None:
        return get_default_vocabulary()
    else:
        return HFSentencePieceVocabulary(vocab_name)



