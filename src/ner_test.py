import os

import random
import torch
from transformers import T5Tokenizer

from models.t5_encoder_models import T5EncoderCRF

NUM_CLASSES = 34

class T5EncoderCRFforNER(object):
    def __init__(self,
            model_dir="../data/t5_encoder_ner",
            model_vocab="spiece.model",
            label_vocab="label.model",
            model_fname="pytorch_model.bin",
            num_classes=NUM_CLASSES,
            ne_formmater="[{entity}]{tag}"):
        
        # ord('▁'): 9601
        # self.white_space_symbol = chr(9601)

        self.ne_formmater = ne_formmater

        self.num_classes = num_classes
        self.tokenizer = T5Tokenizer(os.path.join(model_dir, model_vocab))
        self.label_tokenizer = T5Tokenizer(os.path.join(model_dir, label_vocab))

        self.model = T5EncoderCRF(num_classes, cfg_path=model_dir)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, model_fname)))
        self.model.eval()
    
    def _generate(self, input_str):
        inputs = self.tokenizer.encode_plus("ner: {}".format(input_str), return_tensors="pt")
        sequence_of_tags = self.model(inputs)
        return sequence_of_tags
    
    def generate(self, input_str):
        input_str = input_str.strip()
        outputs = self._generate(input_str)

        sots = outputs[0]
        inputs = self.tokenizer.sp_model.encode(input_str, out_type=str)

        # we add task token for multitask model
        # because we add task token, remove "ner: " tag
        tags = [self.label_tokenizer.decode(x) for x in sots[:-4]]

        input_tag_pairs = list(zip(inputs, tags))

        tag_merged = self.postproc(input_tag_pairs)

        formatted_text = self.formatting(input_str, tag_merged)

        return {
            'text': input_str,
            'tags': tag_merged,
            'fmt_txt': formatted_text
        }
    
    def formatting(self, text, tag_infos):
        formatted_text = text

        offset = 0

        for tag_info in tag_infos:
            tag, tag_pos = tag_info

            entity = text[tag_pos[0]:tag_pos[1]]
            formatted_str = self.ne_formmater.format(entity=entity, tag=tag)

            st, ed = tag_pos[0]+offset, tag_pos[1]+offset
            formatted_text = formatted_text[:st] + formatted_str + formatted_text[ed:]
            offset += len(formatted_str) - len(entity)
        return formatted_text
    
    def postproc(self, input_tag_pairs):
        def unwrap_tag(tag):
            if tag[0] == chr(9601):
                tag_class = tag[1:]
                sp_tag = tag_class.split('-')
                if sp_tag[0] == 'O':
                    return ('O', 'O')
                else:
                    return (sp_tag[1], sp_tag[0])
            else:
                return ('unk', 'unk')
        
        def get_len(tk):
            st = 0
            ed = len(tk)
            if tk[0] == chr(9601):
                st = 1
            return (st, ed)


        base_pos = len(input_tag_pairs[0][0])-1
        tk_pos = [(0, base_pos)]
        for idx, (tk, tag) in enumerate(input_tag_pairs):
            tk_offset = get_len(tk)
            if idx > 0:
                tk_pos.append((tk_offset[0]+base_pos, tk_offset[1]+base_pos))
                base_pos += tk_offset[1]

            input_tag_pairs[idx] = (tk, unwrap_tag(tag))

        input_tag_pos_triples = list(zip(input_tag_pairs, tk_pos))

        # merge tags
        tag_stack = []
        pos_stack = []
        tag_merged = []
        for (tk, tag), pos in input_tag_pos_triples:
            tag_n, tag_a = tag
            if tag_a == 'I':
                tag_stack.append(tag)
                pos_stack.append(pos)
            else:
                if len(tag_stack) > 0:
                    tag_merged.append((tag_stack[0][0], (pos_stack[0][0], pos_stack[-1][1])))
                    tag_stack.clear()
                    pos_stack.clear()
                if tag_a == 'B':
                    tag_stack.append(tag)
                    pos_stack.append(pos)
        if len(tag_stack) > 0:
            tag_merged.append((tag_stack[0][0], (pos_stack[0][0], pos_stack[-1][1])))
            tag_stack.clear()
            pos_stack.clear()
        
        return tag_merged


if __name__ == "__main__":
    ner_model = T5EncoderCRFforNER(model_dir="../data/t5_encoder_ner")

    test_str = "오늘 저녁에 여자친구와 서울에 있는 CGV에서 영화를 보기로 했다."
    print(test_str)
    val = ner_model.generate(test_str)
    print(val['fmt_txt'])






