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

import numpy as np

def _to_eos(sequence, eos_id):
    seq_to_eos = []
    for token in sequence:
        if token == eos_id:
            return seq_to_eos
        else:
            seq_to_eos.append(token)
    return seq_to_eos

def postprocess_for_generator(targets, predictions, tokenizer):
    eos_id = tokenizer.eos_token_id
    targets = [tokenizer.decode(_to_eos(sent, eos_id), skip_special_tokens=True) for sent in targets]
    predictions = [tokenizer.decode(_to_eos(sent, eos_id), skip_special_tokens=True) for sent in predictions]
<<<<<<< HEAD
    return targets, predictions
    

def decode_for_generator(gathered_dict, decode_keys, tokenizer):
    eos_id = tokenizer.eos_token_id
    for k in decode_keys:
        gathered_dict[k] = [tokenizer.decode(_to_eos(sent, eos_id), skip_special_tokens=True) for sent in gathered_dict[k]]

    return gathered_dict
=======
    print(targets[0])
    print(predictions[0])
    return targets, predictions
    

>>>>>>> main

