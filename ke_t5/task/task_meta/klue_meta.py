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

## This file would be replaced after packaging and deploy to PyPI.

_KLUE_TC_CLASSES = [
    'IT과학',  # IT/science
    '경제',  # economy
    '사회',  # society
    '생활문화',  # culture
    '세계',  # world
    '스포츠',  # sports
    '정치',  # politics
    '해당없음'  # OOD(out-of-distribution)
]

_KLUE_NLI_CLASSES = [
    'entailment',
    'neutral',
    'contradiction'
]

_KLUE_NER_TAGS = [
    'DT',  # date
    'LC',  # location
    'OG',  # organization
    'PS',  # person
    'QT',  # quantity
    'TI'  # time
]

_KLUE_NER_IOB2_TAGS = [
    'B-DT',
    'I-DT',
    'B-LC',
    'I-LC',
    'B-OG',
    'I-OG',
    'B-PS',
    'I-PS',
    'B-QT',
    'I-QT',
    'B-TI',
    'I-TI',
    'O'
]



_KLUE_RE_RELATIONS = [
    "no_relation",
    "org:dissolved",
    "org:founded",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:members",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]

_KLUE_RE_ENTITY_TYPE = [
    "PER",
    "ORG",
    "POH",
    "DAT",
    "LOC",
    "NOH"
]

_KLUE_DP_SYNTAX = [
    "NP",  # Noun Phrase
    "VP",  # Verb Phrase
    "AP",  # Adverb Phrase
    "VNP",  # Copula Phrase
    "DP",  # Adnoun Phrase
    "IP",  # Interjection Phrase
    "X",  # Pseudo Phrase
    "L",  # Left Parenthesis and Quotation Mark
    "R"  # Right Parenthesis and Quotation Mark
]
_KLUE_DP_FUNC = [
    "SBJ",  # Subject
    "OBJ",  # Object
    "MOD",  # Noun Modifier
    "AJT",  # Predicate Modifier
    "CMP",  # Complement
    "CNJ",  # Conjunction
]

_KLUE_DP_DEPREL_TAGS = [
    "NP",
    "NP_AJT",
    "VP",
    "NP_SBJ",
    "VP_MOD",
    "NP_OBJ",
    "AP",
    "NP_CNJ",
    "NP_MOD",
    "VNP",
    "DP",
    "VP_AJT",
    "VNP_MOD",
    "NP_CMP",
    "VP_SBJ",
    "VP_CMP",
    "VP_OBJ",
    "VNP_CMP",
    "AP_MOD",
    "X_AJT",
    "VNP_AJT",
    "VP_CNJ",
    "IP",
    "X",
    "VNP_OBJ",
    "X_SBJ",
    "X_OBJ",
    "VNP_SBJ",
    "L",
    "AP_AJT",
    "X_CMP",
    "X_CNJ",
    "X_MOD",
    "AP_CMP",
    "R",
    "VNP_CNJ",
    "AP_SBJ",
    "NP_SVJ"
]

KLUE_META={
    'tc_classes': _KLUE_TC_CLASSES,
    'nli_classes': _KLUE_NLI_CLASSES,
    'ner_tags': _KLUE_NER_TAGS,
    'ner_iob2_tags': _KLUE_NER_IOB2_TAGS,
    're_relations': _KLUE_RE_RELATIONS,
    're_entity_type': _KLUE_RE_ENTITY_TYPE,
    'dp_deprels': _KLUE_DP_DEPREL_TAGS,
    'dp_syntax': _KLUE_DP_SYNTAX,
    'dp_func': _KLUE_DP_FUNC
}