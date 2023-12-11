import math
import statistics
from typing import List
from generate_dataset import MyOwnDataset

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch import nn, FloatTensor, LongTensor
from torch.nn import Dropout, ReLU
from torch.utils.data import TensorDataset

from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def compute_token_char_length(tokens: List[int]) -> List[int]:
    # This code needs to be optimized!!!!
    token_char_lens = []
    for token in  tokens:
        string_token = tokenizer.decode([token], skip_special_tokens=True)
        if string_token[:2] =='##':
            string_token = string_token[2:]
        if string_token[-2:] =='##':
            string_token = string_token[:2]
        token_char_lens.append(len(string_token))
    return token_char_lens

# PositionalEncoding implementation from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
def positional_encoding(positions: FloatTensor, device, d_model=50)->FloatTensor:
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    div_term = div_term.to(device)
    positions = positions.to(device)
    pe = torch.zeros(positions.shape[0], d_model)
    positions = positions.unsqueeze(-1)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.to(device)

class ModifiedBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 768)
        self.dropout = Dropout()
        self.relu = ReLU()
        self.final_linear = nn.Linear(768+6, 384)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the last encoder layer
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
        
    def forward(self, input_ids: LongTensor, token_lens_batch: LongTensor):
        # embedding_output: (batch_size, max_seq_len, vec_size)
        
        embedding_output = self.model.embeddings(input_ids=input_ids)
        token_lengths = torch.stack([positional_encoding(token_lens, token_lens.device, 
                                                         d_model=6) for token_lens in  token_lens_batch])
            # for param in l.parameters():
            #     param.requires_grad = False
        # for param in self.model.encoder.layer[-1].parameters():
        #     param.requires_grad = False
        
        out = self.model.encoder(embedding_output)
        # augmented_input = self.linear(embedding_output)
        # augmented_input = self.relu(augmented_input)
        # augmented_input = self.dropout(augmented_input)
        out = torch.concat((out.last_hidden_state, token_lengths), dim=-1)
        out = self.final_linear(out)
        # out: shape(batch_size, seq_len, vec_size)
        out = torch.sum(out, dim=1)
        return out
    

def vectorise_node_params(input_ids: LongTensor, xs: LongTensor, ys: LongTensor,
                    widths: List[float], heights: List[float], token_lengths: LongTensor, 
                    bert: ModifiedBert):
    ''' Converts each nodes in the graph into a vector of shape (num_nodes, 424).
        (384, ) sentence embedding
        (10, )  Position Embedding of x position of shape
        (10, )  Position Embedding of y position of shape
        (10, )  Position Embedding of width of shape
        (10, )  Position Embedding of height of shape
    '''
    batched_text_encodings: List[FloatTensor] = []
    
    for input_id_batch, token_lens_batch in DataLoader(TensorDataset(input_ids, token_lengths), sampler=None, batch_size=3):
        text_encoding = bert(input_id_batch, token_lens_batch)
        batched_text_encodings.append(text_encoding)

    x_encoding = positional_encoding(xs, xs.device, d_model=10)
    y_encoding = positional_encoding(ys, ys.device, d_model=10)
    width_encoding = positional_encoding(widths, widths.device, d_model=10)
    height_encoding = positional_encoding(heights, heights.device, d_model=10)
    batched_text_encodings = torch.concat(batched_text_encodings).to(batched_text_encodings[0].device)
    return torch.hstack([batched_text_encodings, x_encoding, y_encoding, width_encoding, height_encoding])

class GCN(torch.nn.Module):
    def __init__(self):
        # dataset.num_node_features
        super().__init__()
        self.modified_bert = ModifiedBert()
        self.linear1 = nn.Linear(424, 128)
        # self.sequential1 = nn.Sequential(
        #     *[nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Dropout()]* 7)
        self.sequential1 = nn.Sequential(*[nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout()])
        self.sequential2 = nn.Sequential(
            *[nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout()]*0)
        # self.sequential2 = nn.Sequential()
        # self.conv1 = GCNConv(512, 512)
        # self.conv2 = GCNConv(512, 512)
        self.gcn_modules = nn.ModuleList([GCNConv(64, 64), nn.ReLU(), 
                                          nn.Dropout()]*1)
        # self.gcn_modules = nn.ModuleList([])
        self.final_linear = nn.Linear(64, 4)

    def forward(self, data):
        input_tensor: LongTensor = data.x
        input_ids: LongTensor = input_tensor[:, :512]
        attention_masks: LongTensor = input_tensor[:, 512:512*2]
        x_pos: LongTensor = input_tensor[:, (512*2)+0]
        y_pos: LongTensor = input_tensor[:, (512*2)+1]
        height: LongTensor = input_tensor[:, (512*2)+2]
        width: LongTensor = input_tensor[:, (512*2)+3]
        token_lengths: LongTensor = input_tensor[:, (512*2)+4:]
        
        x = vectorise_node_params(input_ids, x_pos, y_pos, height, width, token_lengths, self.modified_bert)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sequential1(x)  
        gcn_input = x
        for module in self.gcn_modules:
            if isinstance(module, GCNConv):     
                x = module(x, data.edge_index)
                x += gcn_input  # Combine so information from early on will not be forgotten [Improve this by combining them via attention]
                continue
            x= module(x)
            
        x = self.sequential2(x)
        x = self.final_linear(x)
        return F.relu(x)


def train(loader, eval_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    # data = dataset[0].to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    # 200
    val_epochs = []
    val_losses = []
    train_epochs = []
    train_loss = []
    epochs = 50
    grad_accumulation = 4
    for epoch in range(epochs):
        losses = []
        for i, data in enumerate(loader):
            data = data.to(device)
            out = model(data)
            target = data.y.to(device)
            loss = F.mse_loss(out[:, 2:], target[:, 2:]) / grad_accumulation
            loss.backward()
            if i%grad_accumulation == 0:
                print('step')
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss)
            print(loss)    
        train_epochs.append(epoch)
        train_loss.append(float(statistics.mean(map(float, losses))))
        with torch.no_grad():
            model.eval()
            eval_dataset = eval_dataset.to(device)
            val_loss = F.mse_loss(model(eval_dataset)[:, 2:], eval_dataset.y[:, 2:]) / grad_accumulation
            val_epochs.append(epoch)
            val_losses.append(float(val_loss.cpu()))
            model.train()
        print(epoch)
        print(train_loss[-1])
    import matplotlib.pyplot as plt
    # plt.plot(val_epochs[100:], val_losses[100: ], label='Validation')
    # plt.plot(train_epochs[100: ], train_loss[100: ], label='Train')
    plt.plot(val_epochs, val_losses, label='Validation')
    plt.plot(train_epochs, train_loss, label='Train')
    plt.legend()
    plt.show()
    return model

if __name__ == '__main__':
    dataset = MyOwnDataset('/home/sp/Downloads/mindmap_ai/')
    print(len(dataset))
    loader = DataLoader(dataset[:-2], batch_size=3, shuffle=True)
    model: GCN = train(loader, dataset[-2])
    torch.save(model.state_dict(), 'model.torch')