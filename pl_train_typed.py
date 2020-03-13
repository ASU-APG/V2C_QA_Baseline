import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from pytorch_lightning.core import LightningModule
import os
from utils.utils import *
from utils.opts import *
import torch.optim as optim
from model.Model import Model
from model.Model import TypeModel, CapQATypeModel
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
import json
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping


from transformers import AdamW
from collections import OrderedDict

from sklearn.metrics import average_precision_score,precision_recall_curve,label_ranking_loss

def npwhere(values,searchval):
    return np.where(values == searchval)[0]

def create_tensordataset(dataset):
        videos,input_id_list,token_type_id_list,attention_mask_list,targets = [],[],[],[],[]
        limit=240
        typet = []
        feat =[]
        for ix,data in tqdm(enumerate(dataset),"Loading Dataset"):
            data=data[0]
            input_ids,token_type_ids,attention_mask = data["q_inputs"],data["q_token_ids"],data["q_mask"]
            videos.append(data["fc_feats"])
            targets.append(data["target"])
            input_id_list.append(input_ids)
            token_type_id_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
            typet.append(data['type_target'])
            feat.append(data['feat'])
#             if ix==limit:
#                 break
        
        videos = torch.tensor(videos,dtype=torch.float)
        input_ids = torch.tensor(input_id_list,dtype=torch.long)
        token_ids = torch.tensor(token_type_id_list,dtype=torch.long)
        attn_mask = torch.tensor(attention_mask_list,dtype=torch.long)
        targets = torch.tensor(targets,dtype=torch.float)
        typet = torch.tensor(typet,dtype=torch.long)
        feat = torch.tensor(feat,dtype=torch.long)
        
        for t in [videos,input_ids,token_ids,attn_mask,targets]:
            print(t.shape,flush=True)
        return TensorDataset(videos,input_ids,token_ids,attn_mask,typet,targets,feat)

class Net(LightningModule):
    
    def __init__(self,hparams=None,train_loader=None,eval_loader=None):
        super(Net, self).__init__()
        self.hparams = hparams
        
        if hparams is not None:
            self.opts = vars(hparams)
            self.train_loader=train_loader
            self.eval_loader=eval_loader
            
            if not self.hparams.use_pretrained:
                self.model = TypeModel()
            else:
                self.model = CapQATypeModel()
                print("Loading Pretrained Caption Model.")
                model_dict = torch.load('pretrained/RNN_ENCODER_510.pth')
                self.model.load_my_state_dict(model_dict)
        
        self.indexlist = json.load(open("data/indexlist.json"))
        indextensor = torch.cuda.LongTensor(self.indexlist)
        self.mask0 = torch.eq(indextensor,0).float()
        self.mask1 = torch.eq(indextensor,1).float()
        self.mask2 = torch.eq(indextensor,2).float()

        self.mask_cache = {}

        self.logsoftmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.bceloss = nn.BCELoss()
        self.nllloss = nn.NLLLoss()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()   
    
    def set_loaders(self,train_loader=None,eval_loader=None):
        self.train_loader=train_loader
        self.eval_loader=eval_loader
        
    def forward(self,feats,input_ids,token_type_ids,attention_mask):
        return self.model.forward(feats,input_ids,token_type_ids,attention_mask)
    
    def prepare_data(self):
        return
    
    def get_masks(self,mode,batch,batch_idx):
#         key = mode+str(batch_idx)
#         if key in self.mask_cache:
#             return self.mask_cache[key]
        
        mask0 = self.mask0.repeat([batch,1])
        mask1 = self.mask1.repeat([batch,1])
        mask2 = self.mask2.repeat([batch,1])
        return [mask0,mask1,mask2]
#         self.mask_cache[key] = [mask0,mask1,mask2]
#         return self.mask_cache[key]

    def calculatelogits(self,anspreds,typepreds,mode,batch_idx):
        batch = anspreds.size()[0]
        mask0,mask1,mask2 = self.get_masks(mode,batch,batch_idx)
        anspreds0 = anspreds*mask0.cuda()*typepreds.select(dim=1,index=0).reshape([batch,1]).repeat([1,5555])
        anspreds1 = anspreds*mask1.cuda()*typepreds.select(dim=1,index=1).reshape([batch,1]).repeat([1,5555])
        anspreds2 = anspreds*mask2.cuda()*typepreds.select(dim=1,index=2).reshape([batch,1]).repeat([1,5555])
        nanspreds=anspreds0+anspreds1+anspreds2        
        return nanspreds
    
    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader
        
    @pl.data_loader
    def val_dataloader(self):
        return self.eval_loader
    
    @pl.data_loader
    def test_dataloader(self):
        return self.eval_loader
        
    def configure_optimizers(self):
            # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.opts['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opts['learning_rate'], eps=1e-09)        
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
        fc_feats,input_ids,token_type_ids,attention_mask,typet,target,feat = batch

        fc_feats = fc_feats.cuda()
        target = target.cuda()
        typet = typet.cuda()
        input_ids,token_type_ids,attention_mask = input_ids.cuda(),token_type_ids.cuda(),attention_mask.cuda()
        probs,type_probs = self.forward(fc_feats,input_ids,token_type_ids,attention_mask)
        probs = probs.squeeze(1)
        
        probs = self.sigmoid(probs)
        type_logit_soft = self.softmax(type_probs)
    
        probs = self.calculatelogits(probs,type_logit_soft,'train',batch_idx)
        loss_val = self.bceloss(probs, target)
        
        type_probs =  self.logsoftmax(type_probs)
        type_loss_val =  self.nllloss(type_probs,typet)
        
        loss_val =  0.5*loss_val+0.5*type_loss_val
        
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        
        tqdm_dict = {'train_loss': loss_val,'type_loss':type_loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output
    
    #Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    #Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
        
    def qa_accuracy(self,preds,target,k):
        preds = torch.topk(preds,k=k).indices
        mean_prec = 0
        mean_rec = 0
        total_count=0
        for pred,tar in zip(preds.cpu().tolist(),target.cpu().tolist()):
            tar = torch.tensor(tar).nonzero().squeeze()
            tar = tar.cpu().tolist()
            if type(pred) == int:
                pred=[pred]
            if type(tar) == int:
                tar=[tar]
            overlap = set(tar).intersection(set(pred))
            prec = len(overlap)/k
            rec = len(overlap)/len(tar)
            total_count+=1
            mean_prec+=prec
            mean_rec+=rec
        return mean_prec/total_count,mean_rec/total_count
        
    
    def any_acc(self,probs,target):
        preds =(probs>0.5)
        interm=(preds.float()*torch.eq(preds.float(),target.float()))
        interm=torch.any(interm.byte(),dim=1)
        return torch.mean(interm.float())
    
    def validation_step(self,batch,batch_idx):
        fc_feats,input_ids,token_type_ids,attention_mask,typet, target,feat = batch
        fc_feats = fc_feats.cuda()
        target = target.cuda()
        typet = typet.cuda()
        input_ids,token_type_ids,attention_mask = input_ids.cuda(),token_type_ids.cuda(),attention_mask.cuda()

        probs,type_probs = self.forward(fc_feats,input_ids,token_type_ids,attention_mask)
        probs = probs.squeeze(1)

        probs = self.sigmoid(probs)
        type_logit_soft = self.softmax(type_probs)
    
        probs = self.calculatelogits(probs,type_logit_soft,'val',batch_idx)
        loss = self.bceloss(probs, target)
        
        type_probs =  self.logsoftmax(type_probs)
        type_loss_val = self.nllloss(type_probs,typet)
        
        loss =  0.5*loss+0.5*type_loss_val
        
        probs = self.sigmoid(probs)
#         accuracy1 = torch.mean(torch.eq((probs>0.5).float(),target.float()).float())
        accuracy = self.any_acc(probs,target)
        
        label_rank_loss = label_ranking_loss(target.cpu(),probs.cpu())
        
        qa_accuracy = self.qa_accuracy(probs,target,1)
        
        type_accuracy = torch.mean(torch.eq(type_probs.argmax(dim=1),typet).float())
        
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            accuracy = accuracy.unsqueeze(0)
            type_accuracy = type_accuracy.unsqueeze(0)
        
        return {'val_loss':loss, 'r@1':torch.tensor(qa_accuracy[1]).cuda(), 'type_acc':type_accuracy, 'lrloss': torch.tensor(label_rank_loss).cuda(), 'p@1': torch.tensor(qa_accuracy[0]).cuda(), 'probs': probs, 'targets': target , 'types': typet, 'feat': feat}
        
    def validation_end(self,outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        val_tacc_mean = 0
        val_lrloss = 0
        val_qa =0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
            
            val_tacc = output['type_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_tacc = torch.mean(val_tacc)

            val_tacc_mean += val_tacc
            
            if self.trainer.use_dp or self.trainer.use_ddp2:
                output['lrloss'] = torch.mean(output['lrloss'])
                output['p@1'] = torch.mean(output['p@1'])
                output['r@1'] = torch.mean(output['r@1'])
            
            val_lrloss += output['lrloss']
            val_qa += output['p@1']
            val_acc_mean += output['r@1']
            

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        val_tacc_mean /= len(outputs)
        val_lrloss /= len(outputs)
        val_qa /= len(outputs)
        
        tqdm_dict = {'val_loss': val_loss_mean, 'r@1': val_acc_mean,'type_acc':val_tacc_mean,'lrloss':val_lrloss,'p@1':val_qa}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result
        
    def test_step(self,batch,batch_idx):
        return self.validation_step(batch,batch_idx)
    
    def test_end(self,outputs):
        result= self.validation_end(outputs)
        print(result)
        preds = None
        targets = None
        typet = None
        feats = None
        
        for output in outputs:
            logits = output["probs"]
            target = output['targets']
            types = output['types']
            feat = output['feat']
            if preds is None:
                preds = logits.detach().cpu().numpy()
                targets = target.detach().cpu().numpy()
                typet = types.detach().cpu().numpy()
                feats = feat.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                targets = np.append(targets, target.detach().cpu().numpy(), axis=0)
                typet = np.append(typet, types.detach().cpu().numpy(), axis=0)
                feats = np.append(feats,feat.detach().cpu().numpy())
                
        type_acc_map = {}
        for k in [1,3,5]:
            for typex in range(0,21):
                tpreds =[]
                ttargets =[]
                for pred,target,feat in zip(preds,targets,feats):
                    if feat==typex:
                        tpreds.append(pred)
                        ttargets.append(target)
                key = str(typex)+"-"+str(k)
                type_acc_map[key] = self.qa_accuracy(torch.tensor(tpreds,dtype=torch.float),torch.tensor(ttargets,dtype=torch.float),k)
        
        print(type_acc_map)
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

       
    if hparams.evaluate:
        print('Loading from:',hparams.load_checkpoint)
        eval_dataset = create_tensordataset(VideoDataset(opts,'val',use_captions=hparams.use_captions))
        eval_loader=DataLoader(eval_dataset, batch_size=opts['batch_size'], shuffle=False,) 
        pretrained_model = Net.load_from_checkpoint(hparams.load_checkpoint)
        pretrained_model.cuda()
        pretrained_model.set_loaders(eval_loader,eval_loader)
        trainer = pl.Trainer(
            gpus=hparams.gpus,
            distributed_backend=hparams.distributed_backend,
            use_amp=hparams.use_16bit)
        trainer.test(pretrained_model)
    else:
        train_dataset = create_tensordataset(VideoDataset(opts,'train',use_captions=hparams.use_captions))
        eval_dataset = create_tensordataset(VideoDataset(opts,'val',use_captions=hparams.use_captions))
        train_loader=DataLoader(train_dataset, batch_size=opts['batch_size'], shuffle=True,)
        eval_loader=DataLoader(eval_dataset, batch_size=opts['batch_size'], shuffle=False,) 
        model = Net(hparams,train_loader,eval_loader)

        if hparams.seed is not None:
            random.seed(hparams.seed)
            torch.manual_seed(hparams.seed)
            cudnn.deterministic = True


        checkpoint_callback = ModelCheckpoint(
                filepath=hparams.checkpoint_path,
                save_top_k=1,
                verbose=1,
                monitor='p@1',
                mode='max',
                prefix=''
            )

        early_stopping = EarlyStopping(
            monitor='p@1',
            mode='max',
            verbose=True,
            patience=30
        )
        trainer = pl.Trainer(
            default_save_path=hparams.checkpoint_path,
            gpus=hparams.gpus,
            max_epochs=hparams.epochs,
            distributed_backend=hparams.distributed_backend,
            use_amp=hparams.use_16bit,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stopping
        )
        trainer.fit(model)
        trainer.test(model)
        
if __name__ == "__main__":
    net = Net()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = parse_opt()    

    # each LightningModule defines arguments relevant to it
    parser = Net.add_model_specific_args(parser, root_dir)
    hyperparams = parser.parse_args()

    main(hyperparams)
    