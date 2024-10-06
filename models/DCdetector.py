from data_provider.data_factory import data_provider
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from layers.Embed import DataEmbedding
from einops import rearrange, reduce, repeat
from math import sqrt
from layers.StandardNorm import Normalize
from tkinter import _flatten
import numpy as np
import time, os
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads 
        
        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        
        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(B, L, H, -1) 
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(B, L, H, -1) 

        # patch_num
        B, L, M = x_patch_num.shape
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(B, L, H, -1) 
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(B, L, H, -1)
        
        # x_ori
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)
        
        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            values, patch_index,
            attn_mask
        )
        
        return series, prior

class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05, output_attention=False):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index, attn_mask):
                                                 
        # Patch-wise Representation
        B, L, H, E = queries_patch_size.shape #batch_size*channel, patch_num, n_head, d_model/n_head
        scale_patch_size = self.scale or 1. / sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size, keys_patch_size) #batch*ch, nheads, p_num, p_num   
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1)) # B*D_model H N N
        
        # In-patch Representation
        B, L, H, E = queries_patch_num.shape #batch_size*channel, patch_size, n_head, d_model/n_head
        scale_patch_num = self.scale or 1. / sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num, keys_patch_num) #batch*ch, nheads, p_size, p_size 
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1)) # B*D_model H S S 

        # Upsampling
        series_patch_size = repeat(series_patch_size, 'b l m n -> b l (m repeat_m) (n repeat_n)', repeat_m=self.patch_size[patch_index], repeat_n=self.patch_size[patch_index])    
        series_patch_num = series_patch_num.repeat(1,1,self.window_size//self.patch_size[patch_index],self.window_size//self.patch_size[patch_index]) 
        series_patch_size = reduce(series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
        series_patch_num = reduce(series_patch_num, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)


        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return (None)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.win_size = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.task_name = configs.task_name
        self.n_heads = 1
        self.d_model = 256
        self.e_layers = 3
        self.output_attention = True
        self.patch_size = [3,5,7]
        self.channel = 55
        self.d_ff = configs.d_ff
        self.dropout = 0.0
        self.activation = 'gelu'
        
        # Patching List  
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, self.d_model, self.dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, self.d_model, self.dropout))

        self.embedding_window_size = DataEmbedding(self.enc_in, self.d_model, self.dropout)
         
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(self.win_size, self.patch_size, self.channel, False, attention_dropout=self.dropout, output_attention=self.output_attention),
                    self.d_model, self.patch_size, self.channel, self.n_heads, self.win_size)for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)


    def anomaly_detection(self, x):
        B, L, M = x.shape #Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = Normalize(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x, None)
        
        # Mutil-scale Patching Operation 
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size, None)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num, None)
            
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            # import ipdb; ipdb.set_trace()
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))
            
        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        else:
            raise ValueError('Only anomaly detection tasks implemented yet')


class Exp_DCdetector(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _build_model(self):
        model = Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input, None, None, None)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.args.seq_len)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.args.seq_len)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.args.seq_len)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True)
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(train_loader):

                model_optim.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input, None, None, None)
                
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.args.seq_len)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.args.seq_len)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                model_optim.step()

            vali_loss1, vali_loss2 = self.vali(test_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

            
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input, None, None, None)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input, None, None, None)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input, None, None, None)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.args.seq_len)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.seq_len)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)        
        
        # matrix = [self.index]
        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        # for key, value in scores_simple.items():
        #     matrix.append(value)
        #     print('{0:21} : {1:0.4f}'.format(key, value))

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)       

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        
        return accuracy, precision, recall, f_score
