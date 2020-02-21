''' Define the Transformer model '''
from utils.utils import *
from model.Decoder import Decoder
from model.EncoderRNN import EncoderRNN
from pytorch_pretrained_bert import BertModel
from model.transformer.SubLayers import MultiHeadAttention

__author__ = 'Jacob Zhiyuan Fang'


class Model(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, vis_emb=2048, d_model=768, bert_type='bert-base-uncased'):

        super().__init__()

        # Initialize the video encoder
        self.encoder = EncoderRNN(vis_emb, d_model, bidirectional=0)

        # Initialize the textual processing module
        self.bert = BertModel.from_pretrained(bert_type)

        # Cross-model attention module
        self.enc_attn = MultiHeadAttention(n_head=8, d_model=768, d_k=64, d_v=64, dropout=0.1)

        # Classification module
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, vis_feat, gt_token, ca_token1, ca_token2, ca_token3):

        # Batch * 40 * 768
        enc_output, *_ = self.encoder(vis_feat)

        # Batch * 768
        *_, gt_output = self.bert(gt_token)
        *_, ca_output1 = self.bert(ca_token1)
        *_, ca_output2 = self.bert(ca_token2)
        *_, ca_output3 = self.bert(ca_token3)

        # Cross-modal attention layer
        gt_output, _ = self.enc_attn(gt_output.unsqueeze(1), enc_output, enc_output, mask=None)
        ca_output1, _ = self.enc_attn(ca_output1.unsqueeze(1), enc_output, enc_output, mask=None)
        ca_output2, _ = self.enc_attn(ca_output2.unsqueeze(1), enc_output, enc_output, mask=None)
        ca_output3, _ = self.enc_attn(ca_output3.unsqueeze(1), enc_output, enc_output, mask=None)

        # Concatenate all attention layer output for classification
        output = torch.cat((gt_output, ca_output1, ca_output2, ca_output3), 0).squeeze(1)
        prob = self.cls_layer(output)

        # Generate labels for multi-GPU joining steps
        labels = torch.zeros(vis_feat.shape[0]*4).cuda()
        labels[0:vis_feat.shape[0]] = 1

        return prob.squeeze(1), labels

