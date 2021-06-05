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
from transformers import AutoTokenizer

DEFAULT_VOCAB_PATH = "KETI-AIR/ke-t5-small"


def get_default_vocabulary():
    return AutoTokenizer.from_pretrained(DEFAULT_VOCAB_PATH)

@gin.configurable
def get_vocabulary(vocab_name=None):
    if vocab_name is None:
        return get_default_vocabulary()
    else:
        return AutoTokenizer.from_pretrained(vocab_name)