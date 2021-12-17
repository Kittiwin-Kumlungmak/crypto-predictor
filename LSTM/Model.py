import torch.nn as nn
import torch, math
import time

class Transformer(nn.Module):
    def __init__(self,
                d_model,
                nhead = 8,
                num_encode_layer= 3,
                num_decode_layer= 3,
                dim_feedforward=2048,
                len_output = 1,
                dropout= 0.1,
                batch_first= True):
        super(Transformer, self).__init__()

        self.transformer = nn.Transformer(d_model = d_model,
                                            nhead= nhead,
                                            num_encoder_layers= num_encode_layer,
                                            num_decoder_layers= num_decode_layer,
                                            dim_feedforward=dim_feedforward,
                                            dropout= dropout,
                                            batch_first= batch_first)

#         self.encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead,
#                                                         dim_feedforward= dim_feedforward, dropout= dropout,
#                                                         batch_first= batch_first)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer= self.encoder_layer,num_layers= num_encode_layer)
        self.fc1 = nn.Linear(in_features= d_model, out_features= len_output)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model= d_model, nhead= nhead,
        #                                                 dim_feedforward= dim_feedforward, dropout= dropout)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer= self.decoder_layer, num_layers= num_decode_layer)
        # self.init_weight()


    # def init_weight(self):
    #     init_range = 0.1
    #     self.fc1.bias.data.zero_()
    #     self.fc1.weight.data.uniform_(-init_range, init_range)

    # def _gen_subseq_mask(self, size):
    #     mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def forward(self, X, y, y_mask):
        # mask = self._gen_subseq_mask(len(X)).to(device)
        # print(X.size())
        # print(y.size())
        out = self.transformer(X, y, tgt_mask= y_mask)
        # print(X.size())
#         out = self.transformer_encoder(X)
        # print(out.size())
        out = self.fc1(out)
        # print(out.size())
        # print('Out: ', out.size())
        return out

class AdamWarmup:
    def __init__(self, d_model, warmup_steps, optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return (self.d_model**(-0.5)) * min((self.current_step**(-0.5)),(self.current_step/self.warmup_steps**(1.5)))
    
    def step(self):
        self.current_step += 1
        self.lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        #update weights
        self.optimizer.step()
        
        
class LSTM(nn.Module):
    def __init__(self, d_model, hidden_size, num_layer, dropout, output_size, batch_first= True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size= d_model,
                            hidden_size= hidden_size,
                            num_layers= num_layer,
                            dropout= dropout,
                            batch_first= batch_first) #lstm layer
        self.linear = nn.Linear(hidden_size, output_size) #output layer
        
    def forward(self, X, state_in):
        out, state_out = self.lstm(X, state_in)
        out = self.linear(out[:,-1,:])
        return out, state_out