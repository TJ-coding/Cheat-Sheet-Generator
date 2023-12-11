from sentence_transformers import SentenceTransformer
import numpy as np

from generate_dataset import add_noise
from eval import drawio2graph, update_drawio
from models import Graph

if __name__ == '__main__':
    source_drawio_file = 'test.drawio'
    graph: Graph = drawio2graph(source_drawio_file)
    sentbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    xs = add_noise(np.array(graph.xs), 1000)
    ys = add_noise(np.array(graph.ys), 1000)
    widths = add_noise(np.array(graph.widths), 200)
    heights = add_noise(np.array(graph.heights), 200)
    update_drawio(graph.ids, xs.tolist(), ys.tolist(), widths.tolist(), heights.tolist(), 
                  source_drawio_file, 'test2.drawio')