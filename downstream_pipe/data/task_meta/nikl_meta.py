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

_NER_TAGS = [
  'PS', # PERSON
  'LC', # LOCATION
  'OG', # ORGANIZATION
  'DT', # DATE
  'TI', # TIME
  'QT', # QUANTITY
  'AF', # ARTIFACT
  'CV', # CIVILIZATION
  'AM', # ANIMAL
  'PT', # PLANT
  'FD', # STUDY_FIELD
  'TR', # THEORY
  'EV', # EVENT
  'MT', # MATERIAL
  'TM' # TERM
]

_NER_IOB2_TAGS = [
    'O',
    'B-PS', # PERSON
    'I-PS', # PERSON
    'B-LC', # LOCATION
    'I-LC', # LOCATION
    'B-OG', # ORGANIZATION
    'I-OG', # ORGANIZATION
    'B-DT', # DATE
    'I-DT', # DATE
    'B-TI', # TIME
    'I-TI', # TIME
    'B-QT', # QUANTITY
    'I-QT', # QUANTITY
    'B-AF', # ARTIFACT
    'I-AF', # ARTIFACT
    'B-CV', # CIVILIZATION
    'I-CV', # CIVILIZATION
    'B-AM', # ANIMAL
    'I-AM', # ANIMAL
    'B-PT', # PLANT
    'I-PT', # PLANT
    'B-FD', # STUDY_FIELD
    'I-FD', # STUDY_FIELD
    'B-TR', # THEORY
    'I-TR', # THEORY
    'B-EV', # EVENT
    'I-EV', # EVENT
    'B-MT', # MATERIAL
    'I-MT', # MATERIAL
    'B-TM' # TERM
    'I-TM' # TERM
]

_COLA_CLASSES = ["unacceptable", "acceptable"]

NIKL_META={
    'ner_tags': _NER_TAGS,
    'ner_iob2_tabs': _NER_IOB2_TAGS,
    'cola_classes': _COLA_CLASSES
}