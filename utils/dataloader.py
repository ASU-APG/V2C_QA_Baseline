import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


full_feature_map = {
    'int': 0,
    'int_neg': 1,
    'int_act': 2,
    'int_act_neg': 3,
    'int_why': 4,
    'int_yn': 5,
    'int_yn_neg': 6,
    'eff': 7,
    'eff_neg': 8,
    'eff_act': 9,
    'eff_act_neg': 10,
    'eff_how': 11,
    'eff_yn': 12,
    'eff_yn_neg': 13,
    'att': 14,
    'att_neg': 15,
    'att_how': 16,
    'att_how_act': 17,
    'att_yn_act': 18,
    'att_yn': 19,
    'att_yn_neg': 20
}

class VideoDataset(Dataset):

    def __init__(self, opt, mode='train', use_captions=False):
        super(VideoDataset, self).__init__()
        self.mode = mode
        if mode=='train':
            self.v2c_qa_hub = json.load(open(opt['v2c_qa_json']))
        elif mode=='val':
            self.v2c_qa_hub = json.load(open(opt['v2c_qa_val_json']))
        self.feats_dir = opt['feats_dir']
        self.opt = opt
        self.all_labels = json.load(open(opt['labels']))
        
        self.label_map = {}
        for ix,label in enumerate(self.all_labels):
            self.label_map[label]=ix
        self.indexsplit = 30000
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_tok_length=60
        self.use_captions = use_captions
        if use_captions:
            self.max_tok_length=80

        
    def tokenize_text(self, text):
        max_length = self.max_tok_length
        mask_padding_with_zero=True
        pad_token_segment_id=0
        pad_on_left=False
        pad_token=0
        
        inputs = self.tokenizer.encode_plus(text,"",add_special_tokens=True,max_length=self.max_tok_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        
        return input_ids,token_type_ids,attention_mask
    
    def get_feature_type(self,feature):
            if 'int' in feature:
                return 0
            if 'eff' in feature:
                return 1
            if 'att' in feature:
                return 2
            
        
    def __getitem__(self, ix=False):

        annotation = self.v2c_qa_hub[ix]

        # Read out the video_id, question, gt_answer and candidate_answers.
        video_idx = annotation['video_id']
        question = annotation['sent']
        caption = annotation['caption']
    
        typet=self.get_feature_type(annotation['feature'])
        
        feat = full_feature_map[annotation['feature']]
        
        text = question
        if self.use_captions:
            text = caption+ " . " + question
        
        input_ids,token_type_ids,attention_mask=self.tokenize_text(question)
        
        target = [0]*len(self.all_labels)
        
        for label in annotation['label'].items():
            target[self.label_map[label[0]]]=label[1]
        

        # Read out the video features as numpy array
        video_feat = np.load(os.path.join(self.opt['feats_dir'], video_idx+'.npy'))

        # Pack all the video-question-gt_answers-ca_answers in dictionary for returning
        data = {}
        data['fc_feats'] = video_feat
        data['q_inputs'] = input_ids
        data['q_token_ids'] = token_type_ids
        data['q_mask'] = attention_mask
        data['target']  = target
        data['type_target'] = typet
        data['feat'] = feat

        return [data,target]

    def __len__(self):
        return len(self.v2c_qa_hub)

