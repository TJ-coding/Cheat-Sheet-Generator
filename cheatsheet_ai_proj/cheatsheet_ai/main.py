from models import Graph
from eval import drawio2graph, update_drawio
from generate_dataset import graph2data
from train import GCN

import os
import sys

import torch
from torch import FloatTensor
from transformers import BertTokenizer
from torch_geometric.data import Data


def find_new_file_path(source_file_path)->str:
    file_name, extension_name = source_file_path.split('/')[-1].split('.')
    dir_path: str = '/'.join(source_file_path.split('/')[:-1])
    index = 0
    construct_file_path = lambda dir_path, file_name, index, extension_name: f"{dir_path}/{file_name}({index}).{extension_name}"
    while os.path.isfile(construct_file_path(dir_path, file_name, index, extension_name)) == True:
        index += 1
    return construct_file_path(dir_path, file_name, index, extension_name)
    
    
if __name__ == '__main__':
    drawio_file_path: str = input('Please enter drawio file path: ')
    if drawio_file_path[0] == "'" and drawio_file_path[-1] == "'":
        drawio_file_path = drawio_file_path[1:-1]
    graph: Graph = drawio2graph(drawio_file_path)
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data: Data = graph2data(graph.titles, graph.xs, graph.ys, graph.widths, graph.heights, 
                            graph.coo_adjacency_matrix, tokenizer)
    model = GCN()
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load('model.torch'))
        out: FloatTensor = model(data)
        out = out.T
    new_file_path = find_new_file_path(drawio_file_path)
    update_drawio(graph.ids, out[0].tolist(), out[1].tolist(), out[2].tolist(), out[3].tolist(), 
                  drawio_file_path, new_file_path)