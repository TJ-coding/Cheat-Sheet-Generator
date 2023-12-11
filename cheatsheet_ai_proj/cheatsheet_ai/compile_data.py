from models import Graph

import json
from N2G import drawio_diagram

from typing import Dict, NamedTuple, Optional, List, Tuple
import os
import io
import xml.etree.ElementTree as ET
from networkx import adjacency_matrix

import numpy as np

def parse_drawio_as_xml(drawio_file_path: str) -> ET.ElementTree:
    '''Opens a drawio file, and return an xml representation of it.'''
    diagram = drawio_diagram()
    diagram.from_file(drawio_file_path)
    xml_str: str = diagram.dump_xml()
    f = io.StringIO(xml_str)
    tree: ET.ElementTree = ET.parse(f)
    return tree


# def extract_shape_objects_and_edge_mxcells(xml_tree: ET.ElementTree)->Tuple[List[ET.Element], List[ET.Element]]:
#     '''Takes a parsed XML and extract objects that contain shape information and edge information.
#     Returns
#     -------
#     shape_objs: List[ET.Element]
#     edge_mxcells: List[ET.Element]
#     '''
#     root: Optional[ET.Element] = xml_tree.find("diagram/mxGraphModel/root")
#     if root is None:
#         raise ValueError("Could not find 'diagram/mxGraphModel/root' in the xml tree") 
#     objects: List[ET.Element] = root.findall(".//object")
#     shape_objs: List[ET.Element] = []
#     edge_mxcells: List[ET.Element] = []
#     for obj in objects:
#         if 'label' in obj.attrib:
#             shape_objs.append(obj)
#         else:
#             mxcell: Optional[ET.Element] = obj.find('mxCell')
#             if mxcell is None:
#                 raise ValueError(f'Could not find mxCell in the edge object: {obj}')
#             edge_mxcells.append(mxcell)
#     return shape_objs, edge_mxcells

def extract_shape_and_edge_elements(xml_tree: ET.ElementTree)->Tuple[List[ET.Element], List[ET.Element]]:
    elements = xml_tree.findall(f'.//object') + xml_tree.findall(f'.//mxCell')
    edge_elements = []
    shape_elements = [] 
    for element in elements:
        if 'target' in element.attrib or 'source' in element.attrib:
            edge_elements.append(element)
            continue
        shape_elements.append(element)
    return shape_elements, edge_elements

def __get_position_and_dimensions_from_shape_obj(shape_obj: ET.Element)->Tuple[Tuple[float, float], Tuple[float, float]]:
    '''Takes a shape object, returns it's position and dimensions
    Returns
    -------
    position: Tuple[float, float]   Tuple[x, y]
    dimensions: Tuple[float, float] Tuple[width, height]
    '''
    mx_geometry: Optional[ET.Element] = shape_obj.find('.//mxGeometry')
    if mx_geometry is None:
        raise ValueError(f'mxGeometry does not exist in {shape_obj}')
    
    x_pos: float = 0
    y_pos: float = 0
    if 'x' in mx_geometry.attrib:
        x_pos = float(mx_geometry.attrib['x'])
    if 'y' in mx_geometry.attrib:
        y_pos = float(mx_geometry.attrib['y'])

    width: float = 0
    height: float = 0
    if 'width' in mx_geometry.attrib:
        width = float(mx_geometry.attrib['width'])
    if 'height' in mx_geometry.attrib:
        height: float = float(mx_geometry.attrib['height'])
    
    return ((x_pos, y_pos), (width, height))

def extract_shape_params(shape_objs: List[ET.Element])->Tuple[List[str], List[str], List[float], List[float], List[float], List[float]]:
    '''Takes a List that contains shape objects and returns ids, labels, positions and dimensions of the shape.
    Returns
    -------
    shape_ids: List[str]
    shape_labels: List[str]
    xs: List[float]
    ys: List[float]
    widths: List[float]
    heights: List[float]
        
    Examples
    -------
    ([
        'A1c231fa48df3ee393c92dccac4dba5e7d',
        'A29bff8cfaf07b4761bf41ca635cc31223'
        ],
    [
        '3. <b>Report</b> to the <b>Police</b>',
        ''
        ],
    [
        (202.0, 0),
        (600.0, 77.5)
        ],
    [
        (268.0, 60.0),
        (200.0, 60.0)
        ]
    )
    '''
    shape_ids: List[str] = []
    shape_labels: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    width: List[float] = []
    heights: List[float] = []
    # Extract positions and dimensions of the shape
    for obj in shape_objs:
        if 'id' not in obj.attrib or (not 'label' in obj.attrib and not'value' in obj.attrib):
            continue    # SKIP
        shape_ids.append(obj.attrib['id'])
        if 'label' in obj.attrib:
            shape_labels.append(obj.attrib['label'])
        else:
            shape_labels.append(obj.attrib['value'])
        position, dimension = __get_position_and_dimensions_from_shape_obj(obj)
        xs.append(position[0])
        ys.append(position[1])
        width.append(dimension[0])
        heights.append(dimension[1])        
    return shape_ids, shape_labels, xs, ys, width, heights

def extract_edge_params(edge_elements: List[ET.Element])->Tuple[List[str], List[str], List[str], List[List[Tuple[float, float]]]]:
    '''
    Takes a List that contains edge objects and returns edge styles, source_ids, target_ids and edge_points.append
    Note: It does not extract edges that do not contain source or target!!!!
    Returns
    -------
    edge_styles: List[str]
        Contains style information of the edge
    source_ids: List [Str]
        Contains the id of the shape that the edge is pointing from
    target_ids: List[str]
        Contains the id of the shape that the edge is pointing to
    edge_points: List[List[Tuple[float, float]]]] 
        Contains information that define the nodes of the line
        
    Examples
    --------
    ([
        'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;',
        'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0;exitY=0.5;exitDx=0;exitDy=0;'
        ],
    [
        'A1c231fa48df3ee393c92dccac4dba5e7d',
        'A29bff8cfaf07b4761bf41ca635cc31223'
        ],
    [
        'A8410ce43fc9431c902bf5525b19b8010f',
        'A3668c81dc56c1cb5a1bf2a6b1d7410975'
        ],
    [
        [(230, 205), (230, 185)],
        [(230, 205)]
        ],
    )
    '''
    # Filter edge without a source or a target
    edge_elements = list(filter(lambda edge: 'source' in edge.attrib.keys() and 'target' in edge.attrib.keys(), edge_elements))
    
    edge_styles: List[str]= [edge.attrib['style'] if 'style' in edge.attrib else None for edge in edge_elements]
    source_ids: List[str] = [edge.attrib['source'] for edge in edge_elements]
    target_ids: List[str] = [edge.attrib['target'] for edge in edge_elements]
    
    edge_points: List[List[Tuple[float, float]]] = []
    for edge in edge_elements:
        points: List[Tuple[float, float]] = []
        mx_geometry: Optional[ET.Element] = edge.find('mxGeometry')
        if mx_geometry is None:
            points.append([])   # The edge did not contain any points
            continue
        points_array = mx_geometry.find('Array')
        if points_array is None:
            edge_points.append(points)
            continue
        points_objs = points_array.findall('mxPoint')
        for point in points_objs:
            points.append((float(point.attrib['x']), float(point.attrib['y'])))
        edge_points.append(points)
        
    return edge_styles, source_ids, target_ids, edge_points


def generate_adjacency_matrix(ids: List[str], source_ids: List[str], target_ids: List[str])->List[Tuple[int, int]]:
    # Returns adjacency matrix in COO form [Coordinate Form]
    id2index: Dict[str, int] = {ids[i]: i for i in range(len(ids))}
    # Ignore edges with source or target ids that is not found in list
    print(ids)
    print(source_ids)
    print(target_ids)

    filtered = tuple(zip(*filter(lambda ids: ids[0] in id2index and ids[1] in id2index, zip(source_ids, target_ids))))
    source_ids, target_ids = filtered
    #print(len(t))
    #print(type(t))
    source_indices: List[int] = list(map(lambda id: id2index[id], source_ids))
    target_indices: List[int] = list(map(lambda id: id2index[id], target_ids))
    return list(set(list(zip(source_indices, target_indices))+list(zip(target_indices, source_indices))))


def drawio2graph(drawio_file_path: str)->Graph:
    print(drawio_file_path)
    xml: ET.ElementTree = parse_drawio_as_xml(drawio_file_path)
    shape_elements, edge_elements = extract_shape_and_edge_elements(xml)

    shape_ids, shape_titles, xs, ys, widths, heights = extract_shape_params(shape_elements)
    edge_styles, source_ids, target_ids, edge_points = extract_edge_params(edge_elements)
    coo_adjacency_matrix:np.array = generate_adjacency_matrix(shape_ids, source_ids, target_ids)
    return Graph(shape_ids, shape_titles, xs, ys, widths, heights, coo_adjacency_matrix)

def make_dataset():
    dataset_dir: str = 'drawio_dataset'
    drawio_file_names: List[str] = os.listdir(dataset_dir)
    graphs: List[Graph] = [drawio2graph(f'{dataset_dir}/{drawio_file_names[i]}') for i in range(len(drawio_file_names))]
    with open('graph_dataset.json', 'w') as handle:
        json.dump(graphs, handle)
 
if __name__ == '__main__':  
    make_dataset()