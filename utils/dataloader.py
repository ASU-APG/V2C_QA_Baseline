import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def __init__(self, opt, mode='train'):
        super(VideoDataset, self).__init__()
        self.mode = mode
        self.v2c_qa_hub = json.load(open(opt['v2c_qa_json']))
        self.feats_dir = opt['feats_dir']
        self.opt = opt

    def __getitem__(self, ix=False):

        # Note: Modify this part to train/test split for later formal training
        if self.mode == 'train':
            annotation = self.v2c_qa_hub[ix]
        elif self.mode == 'test':
            annotation = self.v2c_qa_hub[ix]

        # Read out the video_id, question, gt_answer and candidate_answers.
        video_idx = annotation['video_id']
        question = annotation['sent']
        gt_answer = list(annotation['label'].keys())[0]
        ca_answer1 = list(annotation['label'].keys())[-3]
        ca_answer2 = list(annotation['label'].keys())[-2]
        ca_answer3 = list(annotation['label'].keys())[-1]

        # Read out the video features as numpy array
        video_feat = np.load(os.path.join(self.opt['feats_dir'], video_idx+'.npy'))

        # Pack all the video-question-gt_answers-ca_answers in dictionary for returning
        data = {}
        data['fc_feats'] = torch.from_numpy(video_feat).type(torch.FloatTensor)
        data['gt_answer'] = question + ' ' + gt_answer
        data['ca_answer1'] = question + ' ' + ca_answer1
        data['ca_answer2'] = question + ' ' + ca_answer2
        data['ca_answer3'] = question + ' ' + ca_answer3

        return data

    def __len__(self):
        return len(self.v2c_qa_hub)
