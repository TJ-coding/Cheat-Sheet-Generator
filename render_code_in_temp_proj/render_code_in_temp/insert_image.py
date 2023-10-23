from typing import Optional, List
import base64
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element

import numpy as np
import cv2

def compress_image(image: np.ndarray, compression_ratio: int = 25) -> str:
    '''Compress the image into a JPEG image and renturn it's binary encoded in base64.
    compression_ratio: int  integer between 0 and 100 (Higher the number, better the quality)
    '''
    result, encoded_image = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, compression_ratio])
    encoded_data = base64.b64encode(encoded_image)
    encoded_data_string = bytes.decode(encoded_data, 'utf-8')
    return encoded_data_string


def insert_image(xml_tree: ElementTree, base64_img: str, width: int, height: int, image_id: str, scale_ratio: float = 0.2) -> ElementTree:
    '''Adds Image to .drawio xml
    scale_ratio: float  factor to scale the size of the image by'''
    image_element = ET.Element('mxCell', id=image_id, value='',
                               style=f'shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/jpeg,{base64_img}', vertex="1", parent="1")
    size_element = ET.Element('mxGeometry', **{'x': '150', 'y': '130', 'width': str(int(
        width*scale_ratio)), 'height': str(int(height*scale_ratio)), 'as': 'geometry'})
    image_element.insert(0, size_element)
    root: Optional[Element] = xml_tree.find('diagram/mxGraphModel/root')
    if root is None:
        raise ValueError('Could not find root tag in the .drawio xml file.')
    root.insert(len(root), image_element)
    return xml_tree


def insert_images_into_drawio(drawio_file_path: str, image_file_paths: List[str]):
    '''Overwrite the drawio file with a .drawio file with compressed images inserted.'''
    tree: ElementTree = ET.parse(drawio_file_path)
    for image_path in image_file_paths:
        image: np.ndarray = cv2.imread(image_path)
        # Image File Name becomes the image id
        image_id: str = image_path.split('/')[-1]
        (height, width) = image.shape[:2]
        base64_image: str = compress_image(image, compression_ratio=25)
        tree = insert_image(tree, base64_image, width, height, image_id, scale_ratio=0.2)
    tree.write(drawio_file_path)
