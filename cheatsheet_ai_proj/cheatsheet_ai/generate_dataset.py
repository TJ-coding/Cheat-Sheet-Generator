import itertools
import json
from typing import Callable, Dict, List, Tuple
from random import choice

from models import Graph

import numpy as np
import torch
from torch import FloatTensor, LongTensor, Tensor
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, InMemoryDataset
from transformers import BertModel, BertTokenizer

from torch.nn import Dropout, ReLU
from torch import nn
    
def bayesian_noise(data: np.array, sigma: float)->np.array:
    noise = np.random.normal(np.zeros(len(data)), sigma)
    scale_noise = np.random.uniform(0, 1)
    noise = scale_noise*noise
    data += noise
    return data

def uniform_noise(data: List[float], spread: float)->np.array:
    noise = -spread/2 - np.random.random(size=len(data))*(spread/2)
    data += noise
    return data

def constant(data: List[float], spread: float)->np.array:
    noise = -spread/2 - np.random.random()*(spread/2)
    data = np.array([noise for i in range(len(data))])
    return data

def unconditionally_random(data: List[float], spread: float)->np.array:
    noise = -spread/2 - np.random.random(size=len(data))*(spread/2)
    return noise

def all_functions_mixed(data: List[float], spread: float)->np.array:
    noise_function = choice([bayesian_noise, uniform_noise, constant, unconditionally_random])
    return noise_function(data, spread)

def noise_node_params(  xs: List[float], ys: List[float], 
                        widths: List[float], heights: List[float])-> Tuple[List[float], List[float], List[float], List[float]]:
    '''Add noise to the positions and the dimensions of the nodes.'''
    noise_functions: List[Callable[[List[float], float], np.array]] = [bayesian_noise, uniform_noise, constant, 
                                                                       unconditionally_random, all_functions_mixed]
    noise_func: Callable[[List[float], float], np.array] = choice(noise_functions)
    xs = noise_func(np.array(xs), 1000)
    ys = noise_func(np.array(ys), 1000)
    widths = noise_func(np.array(widths), 200)
    heights = noise_func(np.array(heights), 200)
    return xs.tolist(), ys.tolist(), widths.tolist(), heights.tolist()

def compute_token_char_length(tokens: List[int], tokenizer: BertTokenizer) -> List[int]:
    # NOTE: This code needs to be optimized!!!!!!!!
    token_char_lens = []
    for token in  tokens:
        string_token = tokenizer.decode([token], skip_special_tokens=True)
        if string_token[:2] =='##':
            string_token = string_token[2:]
        if string_token[-2:] =='##':
            string_token = string_token[:2]
        token_char_lens.append(len(string_token))
    return token_char_lens


def node_params2tensors(titles: List[str], xs: List[float], ys: List[float], 
                        widths: List[float], heights: List[float], tokenizer: BertTokenizer)->Tensor:
    tokens: Dict[str, LongTensor] = tokenizer(titles, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    xs = LongTensor(xs).unsqueeze(-1)
    ys = LongTensor(ys).unsqueeze(-1)
    widths = LongTensor(widths).unsqueeze(-1)
    heights = LongTensor(heights).unsqueeze(-1)
    token_char_lengths: LongTensor = LongTensor([compute_token_char_length(t, tokenizer) for t in tokens['input_ids']])
     
    return torch.hstack([tokens['input_ids'], tokens['attention_mask'], xs, ys, widths, heights, token_char_lengths])

# NOTE: MOVE THIS CODE SOMEWHERE ELSE. IT IS NOT USED HERE MAYBE IT's USED IN EVAL
def graph2data(titles: List[str], xs: np.array, ys: np.array, widths: np.array, 
               heights: np.array, coo_adjacency_matrix: LongTensor, tokenizer)->Data:
    input_tensor = node_params2tensors(titles, xs, ys, widths, heights, tokenizer)
    targets = torch.stack([LongTensor(xs), LongTensor(ys), LongTensor(widths), LongTensor(heights)]).T
    # print('Heights', heights)
    # print('LongTensor', LongTensor(heights))
    # print(targets.shape)
    # What does .t().contiguous() do?
    data: Data = Data(x=input_tensor, edge_index=LongTensor(coo_adjacency_matrix).t().contiguous())
    return data
    
    
def graph2train_data(graph: Graph)->Data:
    # Embed each nodes into vectors
    noised_xs, noised_ys, noised_widths, noised_heights = noise_node_params(graph.xs, graph.ys, graph.widths, graph.heights)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_tensor = node_params2tensors(graph.titles, noised_xs, noised_ys, noised_widths, noised_heights, tokenizer)
    targets = torch.stack([FloatTensor(graph.xs), FloatTensor(graph.ys), FloatTensor(graph.widths), FloatTensor(graph.heights)]).T
    return Data(x=input_tensor, y=targets ,edge_index=LongTensor(graph.coo_adjacency_matrix).t().contiguous())
    
def prepare_data()->List[Data]:
    ''' Convert the data in the graph_dataset.json to a Data instance of the torchG
    
    '''
    # Open Data File
    with open('cheatsheet_ai_proj/graph_dataset.json', 'r') as handle:
        json_data = json.load(handle)
    # Instantiate List of Tuple as a Graph instance
    copies_per_data = 3
    # copies_per_data = 30
    graphs: List[Graph] = itertools.chain(*[[Graph(*d) for d in json_data] for _ in range(copies_per_data)])
    # Convert Graph into Data Format
    data_list: List[Data] = []
    
    for graph in graphs:
        # print('heights', graph.heights)
        train_data: Data = graph2train_data(graph)
        # print('train_height', train_data.y[:, 3])
        data_list.append(train_data)
    return data_list

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into list of Data
        data_list: List[Data] = prepare_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])
        

if __name__ == '__main__':       
    d = MyOwnDataset('cheatsheet_ai_proj/')
    print(d)