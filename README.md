# KE-T5 Downstreams

## Downstreams

Huggingface Echo System을 사용하시는 분들을 위해, 여러 다운스트림 태스크들을 학습시킬 수 있는 모듈을 만들었습니다.
Google의 seqio 일부를 huggingface datasets용으로 만든 `ke_t5.pipe`를 이용하여 명령어 한줄로 task들을 학습시켜볼 수 있습니다.

### Install requirements

모듈 사용에 필요한 패키지들을 설치해줍니다.
```bash
    pip install -r requirements.txt
```

### Type of Model

기본 제공되는 모델들은 크게 2가지로 분류됩니다.
1. Seq2Seq Model (Generative)
2. Encoder Model (BERT like)

Seq2Seq 모델은 입력과 출력이 모두 텍스트이며, Encoder 모델은 Output이 Class logtis(Token level, Sequence level)인 경우가 대부분입니다.

Task의 이름 뒤에 `_gen`이 붙는 경우는 이러한 seq2seq 모델들을 학습시킬 수 있는 task들입니다.
`_gen`이 없는 태스크들은 Encoder 모델에 헤드를 붙여 학습시킬 수 있는 태스크들입니다. (seq2seq으로만 가능한 task의 경우 네이밍 규칙 예외가 존재합니다)


### Training Downstream


지원되는 다운스트림은 ke_t5/task/task.py를 참조하시기 바랍니다.
아래는 KE-T5모델을 이용하여 `nikl_summarization_summary_split`을 학습시키는 경우를 보여줍니다.
(`summarization` task는 generative 모델만 학습니 가능하기 때문에 `_gen`이 붙지 않습니다.)
(**주의** NIKL과 같이 자동으로 다운 받지 못하는 데이터셋은 직접 데이터를 다운받아 압축을 푼 후 루트 디렉토리의 위치를 --hf_data_dir로 입력해줘야 합니다. NIKL 데이터를 준비하는 방법은 [여기](https://github.com/AIRC-KETI/Korean-Copora)를 참고하시기 바랍니다.)

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
    --batch 32 \
    --hf_data_dir "./data" \
    --hf_cache_dir "./cache_dir/huggingface_datasets" \
    --train_split "train[:90%]" \
    --test_split "train[90%:]" \
    --pass_only_model_io true \
    --gin_param="get_dataset.sequence_length={'inputs':512, 'targets':512}" \
    --gin_param="ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'" \
    --pre_trained_model="KETI-AIR/ke-t5-base" \
    --model_name "transformers:T5ForConditionalGeneration" \
    --task 'nikl_summarization_summary_split'
```

**--pass_only_model_io**를 true로 설정하면 모델의 IO로 사용되는 feature로만 mini batch를 만듭니다. 대부분의 generative 모델은 모델의 input과 taget tensor만으로 성능을 측정할 수 있기 때문에 이 값을 true로하면 불필요한 연산을 줄일 수 있습니다. 몇몇 다른 task들의 경우 모델의 input과 target만으로 성능을 측정할 수 없는 경우(NER, extractive QA, etc...)가 있는데, 이 경우에는 이 값을 false로 설정해주어야 합니다. 기본값은 false입니다.

위 예제와 같이 **gin_param**으로 사용할 데이터들의 sequence length와 target length를 입력해주고,
데이터를 preprocessing하는데 사용할 huggingface tokenizer의 이름을 지정해줄 수 있습니다.
이러한 값들을 미리 ***.gin**파일에 입력하여 **--gin_file**로 지정할 수 있습니다.

**gin/train_default.gin** 파일의 경우를 살펴보면, 아래와 같습니다.

```python
get_dataset.sequence_length={'inputs':512, 'targets':512}
ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'

get_optimizer.optimizer_cls=@AdamW
AdamW.lr=1e-3
AdamW.betas=(0.9, 0.999)
AdamW.eps=1e-06
AdamW.weight_decay=1e-2
```


위에서 보듯이 sequence length와 사용할 vocabulary 뿐만 아니라 사용할 optimizer와 파라미터까지 지정되어 있는 것을 확인할 수 있습니다. 이를 이용하여 학습을 시키려면 아래와 같이 입력합니다.


```bash
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
    --batch 32 \
    --gin_file="train_default.gin" \
    --pre_trained_model="KETI-AIR/ke-t5-base" \
    --model_name transformers:T5ForConditionalGeneration \
    --task 'nikl_summarization_summary_split'
```


KE-T5 뿐만 아니라 다른 모델들을 이용해서도 task를 학습할 수 있습니다.
**klue/roberta-small** 모델을 이용하여 klue topic classification을 학습하는 경우 아래와 같이 입력합니다.
이 경우 RoBERTa를 이용한 sequence classification 모델은 transformers 모듈의 RobertaForSequenceClassification 클래스입니다.
`{모듈 경로}:{클래스 이름}` 형태로 **--model_name**에 입력해줍니다. 개인이 만든 모델들도 입력과 출력이 huggingface 모델과 동일하다면 똑같이 학습시킬 수 있습니다.
(gin/klue_roberta_tc.gin에는 vocabulary를 klue/roberta-small의 vocabulary로 설정합니다. classification의 경우 `targets`의 `seqeunce length`는 큰 의미가 없습니다.)


```bash
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
    --batch_size 16 \
    --gin_file="gin/klue_roberta_tc.gin" \
    --pre_trained_model "klue/roberta-small" \
    --model_name transformers:RobertaForSequenceClassification \
    --task 'klue_tc'
```


학습을 더 진행하고 싶다면 **--resume**에 true나 체크포인트 경로를 입력해줍니다. (true의 경우 기본 경로에서 체크포인트를 로드함)
모델은 huggingface `save_pretrained` 합수로 저장하여 나중에 `from_pretrained`로 로딩하려면 저장할 폴더 경로를 **--hf_path**에 입력해줍니다.


```bash
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
    --batch_size 16 \
    --gin_file="gin/klue_roberta_tc.gin" \
    --pre_trained_model "klue/roberta-small" \
    --model_name transformers:RobertaForSequenceClassification \
    --task 'klue_tc' \
    --resume true \
    --hf_path hf_out/klue_bert_tc
```


지금까지의 모든 설명은 distributed setting이었씁니다. **--nproc_per_node**가 학습에 사용할 gpu의 갯수를 말해줍니다.
Single GPU로 학습을 하려면 이 값을 1로 설정하거나, `python -m torch.distributed.launch --nproc_per_node=2` 부분을 `python`으로 바꿔줍니다.

## Test downstream tasks

Test를 할때, generative model의 경우는 huggingface의 beam search, top_p, top_k 등을 위해 generate함수를 사용하고 싶을 수 있습니다.
이 경우 `EvaluationHelper`의 **model_fn**으로 사용하고 싶은 함수 이름을 입력하고,
**model_kwargs**로 함수의 keyword arguments를 입력하면 됩니다. (입력하지 않았을 경우 task에 지정된 kwargs가 입력됩니다.)
**model_input_keys**로 함수에 입력될 데이터의 필드를 정할수 있으며 입력하지 않았을 경우 `input_ids`만 입력됩니다.

**gin/test_default_gen.gin**
```python
get_dataset.sequence_length={'inputs':512, 'targets':512}
ke_t5.task.utils.get_vocabulary.vocab_name='KETI-AIR/ke-t5-base'

EvaluationHelper.model_fn='generate'
EvaluationHelper.model_kwargs={
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
            }
```

```bash

# Test!!!

python -m torch.distributed.launch --nproc_per_node=2 test_ddp.py \
    --gin_file="gin/test_default_gen.gin" \
    --model_name "transformers:T5ForConditionalGeneration" \
    --task 'nikl_summarization_summary_split' \
    --test_split test \
    --resume true
```


### Downstream and Model List

현재 지원되는 다운스트림들의 일부입니다.

| Task 이름 | 형태 |
| --- | --- |
| `klue_tc_gen` | Generative |
| `klue_tc` | Sequence Classification - single_label_classification |
| `klue_nli_gen` | Generative |
| `klue_nli` | Sequence Classification - single_label_classification |
| `klue_sts_gen` | Generative |
| `klue_sts` | Sequence Classification - regression |
| `klue_re` | Sequence Classification - single_label_classification |
| `klue_ner` | Token Classification |
| `nikl_ner` | Token Classification |
| `nikl_ner2020` | Token Classification |
| `nikl_summarization_summary` | Generative |
| `nikl_summarization_topic` | Generative |
| `korquad_gen` | Generative |
| `korquad_gen_context_free` | Generative |
| `kor_3i4k_gen` | Sequence Classification - single_label_classification |
| `kor_3i4k` | Sequence Classification - single_label_classification |


현재 지원되는 T5기반 모델입니다.

| 모델 이름 | 형태 |
| --- | --- |
| `transformers:T5ForConditionalGeneration` | Generative |
| `T5EncoderForSequenceClassificationSimple` | Sequence Classification - single_label_classification |
| `T5EncoderForSequenceClassificationMean` | Sequence Classification - single_label_classification |
| `T5EncoderForTokenClassification` | Token Classification |
| `T5EncoderForEntityRecognitionWithCRF` | Token Classification |


### Custom model

Huggingface model을 상속받아 huggingface output type으로 forward에서 return한다면 이 모델도 사용할 수 있습니다.
예를들어 my_model.py를 다음과 같이 만들었다고 가정합니다. (모델 생성은 ke_t5/models/models.py를 참조해 주세요.)

**my_model_dir/my_model.py**
```python
from transformers import T5EncoderModel
from ke_t5.models.loader import register_model

@register_model("abcdefg")
class MyModel(T5EncoderModel):
    ...

```

위의 `@register_model` 데코레이터는 `MyModel` 클래스를 **abcdefg**라고 등록한다는 것입니다.
따라서 **--model_name**으로 모듈 이름 없이 **abcdefg**를 입력해주면 됩니다.
만약 저 decorator를 붙이지 않았을 경우는 **my_model:abcdefg**로 입력해주면 됩니다.

예를 들어 기본 제공되는 모델들중 `T5EncoderForSequenceClassificationMean` 클래스는 ke_t5.models.models에 위치해 있고,
**T5EncoderForSequenceClassificationMean**으로 이름을 등록했기 때문에,
**T5EncoderForSequenceClassificationMean** 또는 **ke_t5.models.models:T5EncoderForSequenceClassificationMean** 둘 중 아무거나 **--model_name**으로 입력할 수 있습니다.
또한 Camel case로 명명된 Class의 경우 **ke_t5.models.models:t5_encoder_for_sequence_classification_mean**와 같이 대문자마다 `_`를 대신 이용하셔도 됩니다.

custom module을 train_ddp script에서 동작하게 하려면 **--module_import**에 모듈의 경로를 입력해줍니다.
```bash
python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py \
    --batch_size 16 \
    --gin_file="my_model.gin" \
    --pre_trained_model "path_to_pretrained_model_weights" \
    --model_name "abcdefg" \
    --task 'klue_tc' \
    --module_import "my_model_dir.my_model"
```

자신의 모델에 맞는 huggingface vocab path를 입력해주는 것을 잊지마세요.


# Samples

몇가지 샘플 모델을 공유합니다.

| task | model | base model | URL |
| --- | --- | --- | --- |
| `nikl_ner` | `T5EncoderForEntityRecognitionWithCRF` | **KETI-AIR/ke-t5-base** | [Download](https://drive.google.com/file/d/1qJOxWpWgb0nJP8rgsK2PqVY9Cr_jOpXm/view?usp=sharing) |
| `nikl_ner2020` | `T5EncoderForEntityRecognitionWithCRF` | **KETI-AIR/ke-t5-base** | [Download](https://drive.google.com/file/d/1ijN_y26QFwq26BXRYbGHSJFukkZvxOZh/view?usp=sharing) |


Sample code (샘플 모델 중 nikl_ner 모델들을 다운 받았다고 가정)

```python

from transformers import T5Tokenizer
from ke_t5.models import loader, models

model_path = 'path_to_model_directory'
model_name = 'T5EncoderForEntityRecognitionWithCRF'
model_cls = loader.load_model(model_name)

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = model_cls.from_pretrained(model_path)
id2label = model.config.id2label

# 출처 : 경상일보(http://www.ksilbo.co.kr)
# author: 이춘봉기자 bong@ksilbo.co.kr
# URL: http://www.ksilbo.co.kr/news/articleView.html?idxno=903455
input_txt = "울산시설공단은 다양한 꽃·나무 감상 기회를 제공해 시민들의 \
    코로나 블루를 해소하고 이색적인 공간을 연출하기 위해 울산대공원 울산대종 \
    뒤편 야외공연장 상단에 해바라기 정원을 조성했다고 13일 밝혔다."

inputs = tokenizer(input_txt, return_tensors="pt")
output = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )

input_ids = inputs.input_ids[0]
predicted_classes = output.logits[0]
inp_tks = [tokenizer.decode(x) for x in input_ids]
lbls = [id2label[x] for x in predicted_classes]
print(list(zip(inp_tks, lbls)))


# --------------------------------------------------------
## NIKL NER의 경우
[('울산', 'B-OG'), ('시설공단', 'I-OG'), ('은', 'O'), ('다양한', 'O'), 
('꽃', 'B-PT'), ('·', 'O'), ('나무', 'B-PT'), ('감상', 'O'), 
('기회를', 'O'), ('제공해', 'O'), ('시민들의', 'B-CV'), ('코로나', 'O'), 
('블루', 'O'), ('를', 'O'), ('해소하고', 'O'), ('이색적인', 'O'), 
('공간을', 'O'), ('연출', 'O'), ('하기', 'O'), ('위해', 'O'), 
('울산', 'B-LC'), ('대', 'I-LC'), ('공원', 'I-LC'), ('울산', 'B-LC'), 
('대', 'I-LC'), ('종', 'I-LC'), ('뒤편', 'B-TM'), ('야외', 'O'), 
('공연장', 'O'), ('상단', 'O'), ('에', 'O'), ('해바라기', 'B-PT'), 
('정원을', 'O'), ('조성했다', 'O'), ('고', 'O'), ('13', 'B-DT'), 
('일', 'I-DT'), ('밝혔다', 'O'), ('.', 'O'), ('</s>', 'O')]

## NIKL NER 2020의 경우
[('울산', 'B-OGG_POLITICS'), ('시설공단', 'I-OGG_POLITICS'), 
('은', 'O'), ('다양한', 'O'), ('꽃', 'B-PT_PART'), ('·', 'O'), 
('나무', 'O'), ('감상', 'O'), ('기회를', 'O'), ('제공해', 'O'), 
('시민들의', 'O'), ('코로나', 'O'), ('블루', 'O'), ('를', 'O'), 
('해소하고', 'O'), ('이색적인', 'O'), ('공간을', 'O'), ('연출', 'O'), 
('하기', 'O'), ('위해', 'O'), ('울산', 'B-LC_OTHERS'), 
('대', 'I-LC_OTHERS'), ('공원', 'I-LC_OTHERS'), 
('울산', 'B-AF_CULTURAL_ASSET'), ('대', 'I-AF_CULTURAL_ASSET'), 
('종', 'I-AF_CULTURAL_ASSET'), ('뒤편', 'O'), ('야외', 'O'), 
('공연장', 'O'), ('상단', 'O'), ('에', 'O'), ('해바라기', 'B-PT_FLOWER'), 
('정원을', 'O'), ('조성했다', 'O'), ('고', 'O'), ('13', 'B-DT_DAY'), 
('일', 'I-DT_DAY'), ('밝혔다', 'O'), ('.', 'O'), ('</s>', 'O')]
# --------------------------------------------------------
```


## Seq Pipe

TODO Seq pipe에 대하여 설명할 것.

## TODO

- [ ] Seq Pipe 설명 추가
- [ ] Generative model을 위한 Mixture task 추가
- [ ] Coreference Resolution 코드 추가
