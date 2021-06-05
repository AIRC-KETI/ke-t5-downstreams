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





from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('KETI-AIR/ke-t5-small')



test_txt = "정부가 상반기에 전 국민 25%에 해당하는 1,300만 명 이상에게 코로나19 백신을 접종하겠다는 계획을 조기에 달성할 것으로 전망했다. 고령층의 접종 예약률이 80%를 넘김에 따라 1,200만 명에서 1,300만 명으로, 다시 1,300만 명+α로 수정했던 목표치를 또 한번 상향한 것이다. 정부는 '2학기 전면 등교' 에 대비해 유치원, 어린이집 교사와 돌봄인력에게 맞힐 백신도 화이자나 모더나로 변경하고, 접종 시기도 여름방학 기간으로 조정했다."

tokenized_text = tokenizer(test_txt)

print(tokenized_text)
for idx in range(5, 20):
    token_ix = tokenized_text.char_to_token(idx)
    print(token_ix)
