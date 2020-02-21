import os
from utils.utils import *
from utils.opts import *
import torch.optim as optim
from model.Model import Model
from torch.utils.data import DataLoader
from utils.dataloader import VideoDataset
from model.transformer.Optim import ScheduledOptim
from pytorch_pretrained_bert import BertTokenizer

def tokenize_text(tokenizer, list_text):
    """
    :param tokenizer: Bert Tokenizer
    :param list_text: List of the raw texts/
    :return: batch tensor.
    """
    tokens = [torch.LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))) for x in list_text]
    lengths = [len(_) for _ in tokens]

    # Padding the sentences into batch
    batch_tokens = torch.zeros(len(lengths), max(lengths))
    for idx, token in enumerate(tokens):
        batch_tokens[idx, 0:lengths[idx]] = tokens[idx]
    return batch_tokens.long()

def train(loader, model, optimizer, tokenizer, opt):
    # model.train()
    for epoch in range(opt['epochs']):
        iteration = 0

        for data in loader:
            torch.cuda.synchronize()

            # Batch * length * 1024
            fc_feats = data['fc_feats'].cuda()
            gt_answers = data['gt_answer']
            ca_answer1 = data['ca_answer1']
            ca_answer2 = data['ca_answer2']
            ca_answer3 = data['ca_answer3']

            # Pre-process the captions using BERT tokenizer
            gt_tokens = tokenize_text(tokenizer, gt_answers).cuda()
            ca_tokens1 = tokenize_text(tokenizer, ca_answer1).cuda()
            ca_tokens2 = tokenize_text(tokenizer, ca_answer2).cuda()
            ca_tokens3 = tokenize_text(tokenizer, ca_answer3).cuda()

            # Feed into the model for training
            prob, labels = model(fc_feats, gt_tokens, ca_tokens1, ca_tokens2, ca_tokens3)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(prob, labels)

            # Backward loss and clip gradient
            loss.backward()
            optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)

            # update parameters
            loss_item = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            print('iter %d (epoch %d), loss = %.6f' % (iteration, epoch, loss_item))

        if epoch % opt['save_checkpoint_every'] == 0:
            model_path = os.path.join(opt['checkpoint_path'], 'V2C_QA_Bert_Base_%d.pth' % epoch)
            model_info_path = os.path.join(opt['checkpoint_path'], 'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print('model saved to %s' % model_path)

            # Save the logging information
            with open(model_info_path, 'a') as f:
                f.write('model_%d, loss: %.6f' % (epoch, loss_item))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)

    model = Model()
    model = model.cuda()

    # Load pre-trained checkpoint
    # cap_state_dict = torch.load('./save/model_cap-att.pth')
    # model_dict = model.state_dict()
    # model_dict.update(cap_state_dict)
    # model.load_state_dict(model_dict)

    # Initialize the optimizer
    optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09), 512, opt['warm_up_steps'])

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train(dataloader, model, optimizer, tokenizer, opt)

if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    main(opt)