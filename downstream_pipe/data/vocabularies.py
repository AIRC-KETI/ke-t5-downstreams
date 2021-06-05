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

import copy

from seqio.vocabularies import SentencePieceVocabulary
from transformers import AutoTokenizer

class HFSentencePieceVocabulary(SentencePieceVocabulary):
    def __init__(self, vocab_name):
        tokenizer = AutoTokenizer.from_pretrained(vocab_name)

        assert hasattr(tokenizer, 'vocab_file') and hasattr(tokenizer, '_extra_ids'), 'There are no properties vocab_file and extra_ids in tokenizer!'
        vocab_file = tokenizer.vocab_file
        extra_ids = tokenizer._extra_ids

        super().__init__(vocab_file, extra_ids=extra_ids)

# class HFVocabulary(Vocabulary):
#     def __init__(self, vocab_name):
#         self._tokenizer = AutoTokenizer.from_pretrained(vocab_name)

#         _extra_ids = 0
#         if hasattr(self._tokenizer, '_extra_ids'):
#             _extra_ids = self._tokenizer._extra_ids
#         super().__init__(_extra_ids)


