from compile_data import drawio2graph
from models import Graph
from generate_dataset import graph2data
from train import GCN

import xml.etree.ElementTree as ET

import torch
from torch import FloatTensor, LongTensor, norm
from torch_geometric.data import Data
from transformers import BertTokenizer

# def graph2eval_data(graph: Graph, tokenizer: BertTokenizer)->Data:
#     node_vectors: FloatTensor= graph2data(graph.titles, graph.xs, graph.ys, graph.widths, graph.heights, tokenizer)
#     # What does .t().contiguous() do?
#     return Data(x=node_vectors, edge_index=LongTensor(graph.coo_adjacency_matrix).t().contiguous())

def update_graph(graph: Graph, out: FloatTensor)->Graph:
    out = out.T
    graph.xs = out[0].tolist()
    graph.ys = out[1].tolist()
    graph.widths = out[2].tolist()
    graph.heights = out[3].tolist()
    return graph


def update_drawio(ids, xs, ys, widths, heights, drawio_file_path: str, save_path: str):
    with open(drawio_file_path, 'r') as handle:
        tree: ET.ElementTree = ET.parse(handle)    
    for i in range(len(ids)):
        shape_element = tree.find(f".//object[@id='{ids[i]}']/mxCell/mxGeometry")
        if shape_element is None:
            shape_element = tree.find(f".//mxCell[@id='{ids[i]}']/mxGeometry")
        if shape_element is None:
            raise ValueError('Could not find the shape element of the id')

        shape_element.attrib['x'] = str(xs[i])
        shape_element.attrib['y'] = str(ys[i])
        shape_element.attrib['width'] = str(widths[i])
        shape_element.attrib['height'] = str(heights[i])

    tree.write(save_path)

if __name__ == '__main__':
    source_drawio_file = 'cheatsheet_ai_proj/cheatsheet_ai/test.drawio'
    graph: Graph = drawio2graph(source_drawio_file)
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data: Data = graph2data(graph.titles, graph.xs, graph.ys, graph.widths, graph.heights, 
                            graph.coo_adjacency_matrix, tokenizer)
    normalize_width = 140
    normalize_height = 80
    model = GCN()
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load('model.torch'))
        out: FloatTensor = model(data)
        out = out.T
        
    print('x')
    print(out[0].tolist())
    print('y')
    print(out[1].tolist())
    print('width')  
    print(out[2].tolist())
    print('height')
    print(out[3].tolist())
    # update_drawio(graph.ids, out[0].tolist(), out[1].tolist(), out[2].tolist(), out[3].tolist(), 
    #               source_drawio_file, 'cheatsheet_ai_proj/cheatsheet_ai/test2.drawio')
    update_drawio(graph.ids, out[0].tolist(), out[1].tolist(), out[2].tolist(), out[3].tolist(), 
                  source_drawio_file, 'cheatsheet_ai_proj/cheatsheet_ai/test2.drawio')