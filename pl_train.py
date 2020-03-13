from pytorch_lightning.core import LightningModule
import os
from utils.utils import *
from utils.opts import *
import torch.optim as optim
from model.Model import Model
from torch.utils.data import DataLoader
from utils.dataloader import VideoDataset
from model.transformer.Optim import ScheduledOptim
from torch.optim.lr_scheduler import LambdaLR
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn as nn


from transformers import AdamW
from collections import OrderedDict


def create_tensordataset(dataset):
        videos,input_id_list,token_type_id_list,attention_mask_list,targets = [],[],[],[],[]
#         limit=100
        for ix,data in tqdm(enumerate(dataset),"Loading Dataset"):
            data=data[0]
            input_ids,token_type_ids,attention_mask = data["q_inputs"],data["q_token_ids"],data["q_mask"]
            videos.append(data["fc_feats"])
            targets.append(data["target"])
            input_id_list.append(input_ids)
            token_type_id_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
#             if ix==limit:
#                 break
        
        videos = torch.tensor(videos,dtype=torch.float)
        input_ids = torch.tensor(input_id_list,dtype=torch.long)
        token_ids = torch.tensor(token_type_id_list,dtype=torch.long)
        attn_mask = torch.tensor(attention_mask_list,dtype=torch.long)
        targets = torch.tensor(targets,dtype=torch.float)
        
        for t in [videos,input_ids,token_ids,attn_mask,targets]:
            print(t.shape,flush=True)
        return TensorDataset(videos,input_ids,token_ids,attn_mask,targets)

class Net(LightningModule):
    
    def __init__(self,hparams=None,train_loader=None,eval_loader=None):
        super(Net, self).__init__()
        self.hparams = hparams
        self.model = Model()
        
        self.logsoftmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.bceloss = nn.BCELoss()
        self.nllloss = nn.NLLLoss()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        if hparams is not None:
            self.opts = vars(hparams)
            self.train_loader=train_loader
            self.eval_loader=eval_loader
            
        
    def forward(self,feats,input_ids,token_type_ids,attention_mask):
        return self.model.forward(feats,input_ids,token_type_ids,attention_mask)
    
    def prepare_data(self):
        return
    
    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader
        
    @pl.data_loader
    def val_dataloader(self):
        return self.eval_loader
        
    def configure_optimizers(self):
            # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.opts['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opts['learning_rate'], eps=1e-09)
#         optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()),betas=(0.9, 0.98), eps=1e-09)
        
        scheduler = self.get_linear_schedule_with_warmup(optimizer,self.opts['warm_up_steps'],len(self.train_loader)*self.opts['epochs'])
        return [optimizer], [scheduler]
    
    def get_linear_schedule_with_warmup(self,optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)
        
    def training_step(self,batch,batch_idx):
        fc_feats,input_ids,token_type_ids,attention_mask, target = batch

        fc_feats = fc_feats.cuda()
        target = target.cuda()
        input_ids,token_type_ids,attention_mask = input_ids.cuda(),token_type_ids.cuda(),attention_mask.cuda()
        probs = self.forward(fc_feats,input_ids,token_type_ids,attention_mask)
        loss_val = self.bce_loss(probs.squeeze(1), target)
        
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        
        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
        
    def validation_step(self,batch,batch_idx):
        fc_feats,input_ids,token_type_ids,attention_mask, target = batch
        fc_feats = fc_feats.cuda()
        target = target.cuda()
        input_ids,token_type_ids,attention_mask = input_ids.cuda(),token_type_ids.cuda(),attention_mask.cuda()
        probs = self.forward(fc_feats,input_ids,token_type_ids,attention_mask)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probs.squeeze(1), target)
        
        probs = self.sigmoid(probs)
#         accuracy1 = torch.mean(torch.eq((probs>0.5).float(),target.float()).float())
        accuracy = torch.mean(torch.any(((probs>0.5).float()*torch.eq((probs>0.5).float(),target.float()).float()).byte(),dim=1).float())
        
        score, label = probs.max(1)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            accuracy = accuracy.unsqueeze(0)
        
        return {'val_loss':loss,'val_acc':accuracy}
        
    def validation_end(self,outputs):
#         print(outputs,flush=True)
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result
        
        
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        #Avaliable as self.hparams
        return parent_parser
    
    
def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    opts=vars(hparams)
    train_dataset = create_tensordataset(VideoDataset(opts,'train'))
    eval_dataset = create_tensordataset(VideoDataset(opts,'val'))
    
    train_loader=DataLoader(train_dataset, batch_size=opts['batch_size'], shuffle=True,)
    eval_loader=DataLoader(eval_dataset, batch_size=opts['batch_size'], shuffle=False,)

    model = Net(hparams,train_loader,eval_loader)

    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        cudnn.deterministic = True
    trainer = pl.Trainer(
        default_save_path=hparams.checkpoint_path,
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit
    )
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)
        
if __name__ == "__main__":
    net = Net()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = parse_opt()    

    # each LightningModule defines arguments relevant to it
    parser = Net.add_model_specific_args(parser, root_dir)
    hyperparams = parser.parse_args()

    main(hyperparams)
    