from pathlib import Path
import sys

from torch import FloatTensor
sys.path.append('cheatsheet_ai_proj/cheatsheet_ai')

from numpy import diag
test_data_dir = Path(__file__).parent / "cheatsheet_ai_proj/cheatsheet_ai"
from typing import Tuple
from cheatsheet_ai.compile_data import drawio2graph
from cheatsheet_ai.models import Graph
from cheatsheet_ai.eval import update_drawio

from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET



import unittest

class TestDrawio2Graph(unittest.TestCase):
    def initialise_diagram()->Tuple[ET.ElementTree, ET.Element]:
        # Two mxCell with id 0 and 1 must first be initialised for objects to be displayed
        diagram = ET.fromstring('<mxfile><diagram><mxGraphModel><root> \
        <mxCell id="0"/> \
        <mxCell id="1" parent="0"/> \
        </root></mxGraphModel></diagram></mxfile>')
        root = diagram.find('.//root')
        return ET.ElementTree(diagram), root

    @staticmethod
    def make_shape_element(id, label, x, y, width, height):
        if id == 0 or id == 1:
            raise ValueError('id of 0 or 1 is reserve and not allowed')
        shape_xml = f'''<object label="{escape(label).replace('"', "&quot;")}" id="{id}">
        <mxCell style="rounded=0;whiteSpace=wrap;html=1;" vertex="1" parent="1">
        <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>
        </mxCell>
        </object>'''
        return ET.fromstring(shape_xml)

    @staticmethod
    def make_arc_element(id, source_id, target_id):
        if id == 0 or id == 1:
            raise ValueError('id of 0 or 1 is reserve and not allowed')
        edge_xml = f'''
        <mxCell id="{id}" style="edgeStyle=none;html=1;" parent="1" source="{source_id}" target="{target_id}" edge="1">
            <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        
        '''
        return ET.fromstring(edge_xml)
    
    @unittest.SkipTest
    def test_drawio2graph(self):
        source_file_path = 'cheatsheet_ai_proj/drawio_dataset/driving school 2 - 16 planning your route.drawio',
        save_path = 'cheatsheet_ai_proj/cheatsheet_ai/test3.drawio'
        
        graph: Graph = drawio2graph(source_file_path)
        diagram, root = TestDrawio2Graph.initialise_diagram()
        for i in range(len(graph.ids)):
            print(graph.titles[i])
            # if graph.xs[i] == 0:
            #     raise ValueError('The x value was 0')
            # if graph.ys[i] == 0:
            #     raise ValueError('The y value was 0') 
            if graph.widths[i] == 0:
                raise ValueError('The width value was 0') 
            if graph.heights[i] == 0:
                raise ValueError('The height value was 0') 
            root.append(TestDrawio2Graph.make_shape_element(graph.ids[i], graph.titles[i], graph.xs[i], graph.ys[i], graph.widths[i], graph.heights[i]))
        for i in range(graph.coo_adjacency_matrix.shape[1]):
            source_id = graph.ids[graph.coo_adjacency_matrix[0, i]]
            target_id = graph.ids[graph.coo_adjacency_matrix[1, i]]
            root.append(TestDrawio2Graph.make_arc_element(source_id+target_id, source_id, target_id))
        print(ET.dump(diagram))
        diagram.write(save_path)


from cheatsheet_ai.train import compute_edge_vectors
class TestComputeEdgeVectors(unittest.TestCase):
    def test_compute_edge_vectors(self):
        '''
            A       A[0, 0], B[-2, 3] C[10, 5]
           / \    
          B   C
        '''
        node_x_components = FloatTensor([0, -2, 10])
        node_y_components = FloatTensor([0, 3, 5])
        undirected_adjacency_matrix = FloatTensor([[0,1,1],[1,0,0],[1,0,0]])
        edge_vectors = compute_edge_vectors(node_x_components, node_y_components, undirected_adjacency_matrix)
        self.assertListEqual(edge_vectors.tolist(), [[[0,0], [-2, 3], [10, 5]], 
                                                     [[2, -3], [0,0], [0,0]], 
                                                     [[-10, -5], [0,0], [0,0]]])
        
                
if __name__ == '__main__':
    unittest.main()