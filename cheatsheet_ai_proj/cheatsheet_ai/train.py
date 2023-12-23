# import ipdb; ipdb.set_trace()
import itertools
import math
import random
import statistics
from typing import List, Optional

from cv2 import normalize
from sympy import Float
from generate_dataset import MyOwnDataset


import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_torch_sparse_tensor
from torch import Value, nn, FloatTensor, LongTensor
from torch.nn import Dropout, ReLU
from torch.utils.data import TensorDataset
from sklearn import preprocessing
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
        self.normalization_layer = nn.LayerNorm(384)
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the last encoder layer
        # for param in self.model.encoder.layer[1].parameters():
        #     param.requires_grad = True
        for param in self.model.encoder.layer[-1].parameters():
            param.requires_grad = True
        # for param in self.model.encoder.layer[-2].parameters():
        #     param.requires_grad = True
        # for param in self.model.encoder.layer[1].parameters():
        #     param.requires_grad = True
        self.pooling_transformer = torch.nn.Transformer(d_model=768,num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=768, 
                                                        dropout=0.1, layer_norm_eps=1e-05, bias=True,)
        
    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, token_lens_batch: LongTensor):
        # embedding_output: (batch_size, max_seq_len, vec_size)
        # if False in [0 in input_ids[i] for i in range(len(input_ids))]:
        #     breakpoint()
        embedding_output = self.model.embeddings(input_ids=input_ids)
        token_lengths = torch.stack([positional_encoding(token_lens, token_lens.device, 
                                                         d_model=768) for token_lens in  token_lens_batch])
            # for param in l.parameters():
            #     param.requires_grad = False
        # for param in self.model.encoder.layer[-1].parameters():
        #     param.requires_grad = False
        
        extended_attention_mask: torch.Tensor = self.model.get_extended_attention_mask(attention_mask, input_ids.size())
        # WHERE NAN STARTS TO OCCURE
        # with torch.no_grad():
        out = self.model.encoder(embedding_output, attention_mask=extended_attention_mask)
        # augmented_input = self.linear(embedding_output)
        # augmented_input = self.relu(augmented_input)
        # augmented_input = self.dropout(augmented_input)
        
        
        # out = torch.concat((out.last_hidden_state, token_lengths), dim=-1)
        # out = self.final_linear(out)
        # out: shape(batch_size, seq_len, vec_size)
        
        # out = attention_mask.unsqueeze(-1) * out        
        batch_size = out.last_hidden_state.shape[0]
        bert_output = out.last_hidden_state.movedim(0,1)
        attention_mask = attention_mask.to(torch.bool)
        attention_mask = torch.logical_not(attention_mask)  # 1 for padding 0 for non padding
        token_lengths = token_lengths.movedim(0,1)
        out = self.pooling_transformer(bert_output, token_lengths, 
                                       src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        
        out = out[0, :]
        # out = torch.sum(out.last_hidden_state, dim=1) # NOTE: Take into account of the attention mask, you might need to normalize
        # Normalize
        # num_of_non_pad_tokens = torch.sum(attention_mask, dim=1)
        # out = out / num_of_non_pad_tokens.unsqueeze(-1)
        # out = self.normalization_layer(out)    
        if True in [torch.isnan(i) for i in out[:, 0]]:
            breakpoint()
        return out
    

def vectorise_node_params(input_ids: LongTensor, attention_masks: LongTensor, xs: LongTensor, ys: LongTensor,
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
    
    for input_id_batch, attention_mask, token_lens_batch in DataLoader(TensorDataset(input_ids, attention_masks, token_lengths), sampler=None, batch_size=3):
        text_encoding = bert(input_id_batch, attention_mask, token_lens_batch)
        batched_text_encodings.append(text_encoding)

    x_encoding = positional_encoding(xs, xs.device, d_model=10)
    y_encoding = positional_encoding(ys, ys.device, d_model=10)
    width_encoding = positional_encoding(widths, widths.device, d_model=10)
    height_encoding = positional_encoding(heights, heights.device, d_model=10)

    batched_text_encodings = torch.concat(batched_text_encodings).to(batched_text_encodings[0].device)
    # print(batched_text_encodings)
    if True in [torch.isnan(i)for i in batched_text_encodings[:,0]]:
        print('batched_text_encodings has nan')
        breakpoint()
    return torch.hstack([batched_text_encodings, x_encoding, y_encoding, width_encoding, height_encoding])

class GCN(torch.nn.Module):
    def __init__(self, means: Optional[FloatTensor]=None, stds: Optional[FloatTensor]=None):
        super().__init__()
        self.means = torch.nn.Parameter(FloatTensor(4, ))
        self.stds = torch.nn.Parameter(FloatTensor(4, ))
        if not means is None:
            self.means = torch.nn.Parameter(means, requires_grad=False)
        if not stds is None:
            self.stds = torch.nn.Parameter(stds, requires_grad=False)
        self.means.requires_grad = False
        self.stds.requires_grad = False

        # dataset.num_node_features
        self.modified_bert = ModifiedBert()
        self.linear1 = nn.Sequential(nn.Linear(768+40, 384), nn.ReLU(),
            nn.Dropout(), nn.Linear(384, 192), nn.ReLU(),
            nn.Dropout(), nn.Linear(192, 64), nn.ReLU(),
            nn.Dropout())
        # self.linear1 = nn.Linear(1, 64)
        # self.sequential1 = nn.Sequential(
        #     *[nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Dropout()]* 7)
        self.sequential1 = nn.Sequential(*[nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout()]*1)
        self.sequential2 = nn.Sequential(
            *[nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout()]*1)
        # self.sequential2 = nn.Sequential()
        # self.conv1 = GCNConv(512, 512)
        # self.conv2 = GCNConv(512, 512)
        # self.gcn_modules = nn.ModuleList([GCNConv(64, 64), nn.ReLU(), 
        #                                   nn.Dropout(), nn.Linear(64,64), nn.ReLU(), 
        #                                   nn.Dropout()]*4)  # NOTE: Set this to 1 or more!!!
        self.gcn_modules = nn.ModuleList([GCNConv(64, 64), nn.ReLU(), 
                                          nn.Dropout(), nn.Linear(64,64), nn.ReLU(), 
                                          nn.Dropout(), nn.Linear(64,64), nn.ReLU(), 
                                          nn.Dropout(), nn.Linear(64,64), nn.ReLU(), 
                                          nn.Dropout()])
        # self.gcn_modules = nn.ModuleList([])
        self.final_linear = nn.Linear(64, 4)

    def forward(self, data):
        # print(data.x)
        input_tensor: LongTensor = data.x
        input_ids: LongTensor = input_tensor[:, :512]
        attention_masks: LongTensor = input_tensor[:, 512:512*2]
        x_pos: LongTensor = input_tensor[:, (512*2)+0]
        y_pos: LongTensor = input_tensor[:, (512*2)+1]
        height: LongTensor = input_tensor[:, (512*2)+2]
        width: LongTensor = input_tensor[:, (512*2)+3]
        token_lengths: LongTensor = input_tensor[:, (512*2)+4:]
        
        x = vectorise_node_params(input_ids, attention_masks, x_pos, y_pos, height, width, token_lengths, self.modified_bert)
        # print(x.shape)
        # x = torch.FloatTensor([[0], [0.25], [0.5], [0.75], [1]]*(data.x.shape[0]//5)).to(input_tensor.device)
        # print(x.shape)
        if True in [torch.isnan(i)for i in x[:,0]]:
                breakpoint()
        x = self.linear1(x)
        
        # x = self.linear1(torch.vstack([x_pos.to(torch.float).squeeze(-1), 
        #                                y_pos.to(torch.float).squeeze(-1), 
        #                                height.to(torch.float).squeeze(-1), 
        #                                width.to(torch.float).squeeze(-1)]).T)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sequential1(x)  
        gcn_input = x
        # for module in self.gcn_modules:
        #     if isinstance(module, GCNConv):
        #         x = module(x, data.edge_index)
        #         x += gcn_input  # Combine so information from early on will not be forgotten [Improve this by combining them via attention]
        #         continue
        #     x= module(x)
        
        for i in range(20):
            for module in self.gcn_modules:
                if isinstance(module, GCNConv):
                    x = module(x, data.edge_index)
                    x += gcn_input  # Combine so information from early on will not be forgotten [Improve this by combining them via attention]
                    continue
                x= module(x)
            
        x = self.sequential2(x)
        x = self.final_linear(x)
        x *= self.stds
        x += self.means
        return x



def compute_component_residual(node_vectors_components: torch.FloatTensor, 
                               undirected_adjacency_matrix: torch.FloatTensor ):
    '''Computes a matrix of residual between connected nodes.
    I.e. if node_vector_components = [x, y, z]
    adjacency_matrix = [[0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 0]]
    I.e.
        x
       / \
      y   z
    Then Returns:
    [ [0     , x-y   , x-z   ],
      [y-x   , 0     , 0     ],
      [y -z  , 0     , 0    ]]
    '''
    num_nodes = len(node_vectors_components)
    identity_matrix = torch.eye(num_nodes).to(node_vectors_components.device)
    identity_matrix.requires_grad = False
    undirected_adjacency_matrix.requires_grad = False
    A = node_vectors_components * identity_matrix
    # NOTE: You need to take a Transpose if you are taking a Directed Adjacency Matrix
    # Transpose of an undirected adjacency matrix as the same
    B = undirected_adjacency_matrix
    C = torch.sparse.mm(A, B)
    D = node_vectors_components.repeat(num_nodes, 1)*B
    return  (-C) + D

def compute_edge_vectors(node_x_components: torch.FloatTensor, 
                            node_y_components: torch.FloatTensor, 
                            undirected_adjacency_matrix: torch.LongTensor):
    '''Returns Residual Vector of connected nodes
    E.g. if nodes = [(A_x, A_y), (B_x, B_y), (C_x, C_y)]
        adjacency_matrix = [[0, 1, 1],
                            [1, 0, 0],
                            [1, 0, 0]]
    I.e.
        x
       / \
      y   z
       Then Returns:
    [ [(0,0)                , (A_x-B_x, A_y-B_y)    , (A_x-C_x, A_y-C_y)    ],
      [(B_x-A_x, B_y-A_y)   , 0                     , 0                     ],
      [(C_x-A_x, C_y-A_y)   , 0                     , 0                     ]]
    '''
    if isinstance(undirected_adjacency_matrix, torch.LongTensor): # Cast to Float
      undirected_adjacency_matrix = undirected_adjacency_matrix.to(torch.float)
    x_component_residual_matrix = compute_component_residual(node_x_components, undirected_adjacency_matrix)
    y_component_residual_matrix = compute_component_residual(node_y_components, undirected_adjacency_matrix)
    residual_vector_matrix = torch.stack((x_component_residual_matrix, y_component_residual_matrix), dim=-1)
    return residual_vector_matrix

def compute_loss(model_output: FloatTensor, target_output: FloatTensor, coo_edge_index: LongTensor,  
                 means: FloatTensor, stds: FloatTensor)->FloatTensor:
    means.requires_grad = False
    stds.requires_grad = False
    device = model_output.device
    predicted_x = model_output[:, 0]
    predicted_y = model_output[:, 1]
    predicted_width = model_output[:, 2]
    predicted_height = model_output[:, 3]
    target_x = target_output[:, 0]
    target_y = target_output[:, 1]
    sparse_adjacency_matrix: torch.Tensor = to_torch_sparse_tensor(coo_edge_index, size=len(target_x))
    # print(predicted_x.shape)
    # predicted_edge_vectors = compute_edge_vectors(predicted_x, predicted_y, sparse_adjacency_matrix)
    # target_edge_vectors = compute_edge_vectors(target_x, target_y, sparse_adjacency_matrix)
    # print('zero edge_vector count: ', target_edge_vectors.detach()[sparse_adjacency_matrix.to(torch.bool).to_dense()].to('cpu').flatten().tolist().count(0)/ \
    #     len(target_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()].flatten()))
    # if target_edge_vectors.detach()[sparse_adjacency_matrix.to(torch.bool).to_dense()].to('cpu').flatten().tolist().count(0)/ \
    #     len(target_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()].flatten()) > 0.7:
    #     # breakpoint()
    #     print(target_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()])
    
    # print('zero height count: ', target_output[:,2].tolist().count(0)/len(predicted_height))
    # print('zero width count: ', target_output[:, 3].tolist().count(0)/len(predicted_height))
    # target_edge_vectors.requires_grad = False
    
    # Cosine similarity between the predicted edge vectors and the target edge vectors
    '''cosine_similarities = F.cosine_similarity(predicted_edge_vectors, target_edge_vectors, dim=2)[sparse_adjacency_matrix.to(torch.bool).to_dense()]''' 
    '''edge_vector_cosine_loss = F.mse_loss(torch.ones(len(cosine_similarities)).to(device), cosine_similarities)   # Very small between zero and one'''
    
    '''target_edge_vector_magnitudes = torch.norm(target_edge_vectors, dim=2)[sparse_adjacency_matrix.to(torch.bool).to_dense()] '''
    '''predicted_edge_vectors_magnitudes = torch.norm(predicted_edge_vectors, dim=2)[sparse_adjacency_matrix.to(torch.bool).to_dense()]'''
    # Percentage error of edge vector magnitudes
    #magnitude_matrix = (predicted_edge_vectors_magnitudes-target_edge_vector_magnitudes) /(target_edge_vector_magnitudes+1) 
    '''magnitude_matrix = (predicted_edge_vectors_magnitudes-target_edge_vector_magnitudes)'''
    # MSE loss
    '''edge_vector_magnitude_loss =  F.mse_loss(torch.zeros(len(magnitude_matrix)).to(device), magnitude_matrix)'''
    # edge_vector_loss = F.mse_loss(predicted_edge_vectors.to_dense(), target_edge_vectors.to_dense())
    
    #rect_size_loss = F.mse_loss(model_output[:, 2:], target_output[:, 2:])/(torch.mean(torch.abs(target_output[:, 2:]))**2)
    # rect_size_loss = F.mse_loss(torch.ones(model_output[:, 2:].shape).to('cuda'), (model_output[:, 2:]-target_output[:, 2:])/(target_output[:, 2:]+torch.ones(target_output[:, 2:].shape).to('cuda')))
    # rect_size_loss = F.mse_loss(model_output[:, 2:], target_output[:, 2:])
    #return rect_size_loss/100 + edge_vector_cosine_loss*10 + edge_vector_magnitude_loss/100
    # print('target output')
    # print(target_output[:, 2:])
    
    # edge_vector_loss = F.mse_loss(predicted_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()], target_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()])
    # position_loss = (F.mse_loss(target_x, predicted_x) + F.mse_loss(target_y, predicted_y))/2
    
    # print(rect_size_loss*10, edge_vector_loss, position_loss/100)
    # return rect_size_loss*10 + edge_vector_loss + position_loss/100
    # return rect_size_loss + position_loss/100
    # return rect_size_loss + position_loss

    means = means.to(model_output.device)
    stds = stds.to(model_output.device)
    normalized_output = (model_output-means)/stds
    normalized_target = (target_output-means)/stds
    if torch.isnan(model_output[:, 2][0]):
        breakpoint()
    if random.randint(0,50) == 1:
        print('---------------------------')
        print(model_output[:, 2].tolist())
        print(target_output[:, 2].tolist())
        print('---------------------------')
        print(normalized_output[:, 2].tolist())
        print(normalized_target[:, 2].tolist())
        
    predicted_x = normalized_output[:, 0]
    predicted_y = normalized_output[:, 1]
    target_x = normalized_target[:, 0]
    target_y = normalized_target[:, 1]
    # predicted_edge_vectors = compute_edge_vectors(predicted_x, predicted_y, sparse_adjacency_matrix)
    # target_edge_vectors = compute_edge_vectors(target_x, target_y, sparse_adjacency_matrix)
    # edge_vector_loss = F.mse_loss(predicted_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()], target_edge_vectors[sparse_adjacency_matrix.to(torch.bool).to_dense()])
    
    # return (F.mse_loss( normalized_output, normalized_target) + edge_vector_loss)/2
    return F.mse_loss( normalized_output[:, 2:], normalized_target[:, 2:]) + F.mse_loss( normalized_output[:, 1]*normalized_output[:, 2], normalized_target[:, 1]*normalized_target[:, 2]) 

def train(loader, eval_dataset, means: FloatTensor, stds: FloatTensor):
    means.requires_grad = False
    stds.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(means, stds).to(device)
    # data = dataset[0].to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=5e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # epochs = 50*4
    # epochs = 50*2
    # epochs = 50*1
    epochs = 10
    # grad_accumulation = 4
    grad_accumulation = 4
    lr_scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(loader)/grad_accumulation)+1, epochs=epochs)
    model.train()
    # 200
    val_epochs = []
    val_losses = []
    train_epochs = []
    train_loss = []
    # epochs = 10
    for epoch in range(epochs):
        losses = []
        for i, data in enumerate(loader):
            data = data.to(device)
            out = model(data)
            if True in [torch.isnan(i)for i in out[:,0]]:
                breakpoint()
            target = data.y.to(device)
            if torch.isnan(out[:, 2][0]):
                breakpoint()
            # loss = F.mse_loss(out[:, 2:], target[:, 2:]) / grad_accumulation
            loss = compute_loss(out, target, data.edge_index, means, stds) / grad_accumulation
            # loss = F.mse_loss(out, target) / grad_accumulation
            loss.backward()
            if torch.isnan(loss):
                breakpoint()
            if i%grad_accumulation == 0:
                print('step')
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            losses.append(loss)
            print(loss)    
        train_epochs.append(epoch)
        train_loss.append(float(statistics.mean(map(float, losses))))
        with torch.no_grad():
            model.eval()
            eval_dataset = eval_dataset.to(device)
            val_loss = compute_loss(model(eval_dataset), eval_dataset.y, eval_dataset.edge_index, means, stds)/grad_accumulation
            val_epochs.append(epoch)
            val_losses.append(float(val_loss.cpu()))
            model.train()
        print(epoch)
        print(train_loss[-1])  
        if epoch % 25==0:
            print(out[:, -1].tolist())
    import matplotlib.pyplot as plt
    # plt.plot(val_epochs[100:], val_losses[100: ], label='Validation')
    # plt.plot(train_epochs[100: ], train_loss[100: ], label='Train')
    plt.plot(val_epochs, val_losses, label='Validation')
    plt.plot(train_epochs, train_loss, label='Train')
    plt.legend()
    plt.show()
    return model

def data_normalizer(data: LongTensor):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    print(scaler.transform(data))
    return scaler

if __name__ == '__main__':
    dataset = MyOwnDataset('cheatsheet_ai_proj/')
    scaler = data_normalizer(dataset[:].y)
    print(len(dataset))
    # loader = DataLoader(dataset[:-2], batch_size=3, shuffle=True)
    loader = DataLoader(dataset[:-2], batch_size=1, shuffle=True)
    model: GCN = train(loader, dataset[-2], means=FloatTensor(scaler.mean_), stds=torch.sqrt(FloatTensor(scaler.var_)))
    torch.save(model.state_dict(), 'model.torch')