from typing import List, Dict, Optional, Tuple
import hashlib
import copy

from .dom_parser import DocumentDOM, SectionDOM, SubSecDOM, TextGroupDOM

from N2G import drawio_diagram


class DrawioIdGenerator:
    def __init__(self) -> None:
        self.next_id = 0

    def make_id(self, label: str) -> str:
        md5 = hashlib.md5()
        md5.update(label.encode())
        # 'A' Ensures that it starts with char not number
        # Required for layout to work for some reason
        self.next_id += 1
        return f'A{self.next_id}{md5.hexdigest()}'

def escape_for_xml(str_xml: str) -> str:
    '''Escapes special chars for use in XML parsing 
    [N2G automatically parses everything as an xml for some reason]'''
    str_xml = str_xml.replace("&", "&amp;")
    str_xml = str_xml.replace("<", "&lt;")
    str_xml = str_xml.replace(">", "&gt;")
    str_xml = str_xml.replace("\"", "&quot;")
    str_xml = str_xml.replace("'", "&apos;")
    return str_xml

def draw_nodes(labels: List[str],
               positions: List[Tuple[float, float]],
               diagram: drawio_diagram,
               drawio_id_generator: DrawioIdGenerator,
               block_width=135,
               block_height=75) -> tuple[List[str], drawio_diagram]:
    '''Draws nodes of labels. Returns the ids of the nodes and the diagram.'''
    diagram.add_diagram('Page-1')
    ids: List[str] = []
    for i in range(len(labels)):
        label_id: str = drawio_id_generator.make_id(labels[i])
        diagram.add_node(id=label_id, label=escape_for_xml(labels[i]),
                         x_pos=int(positions[i][0]*block_width),
                         y_pos=int(positions[i][1]*block_height),
                         style='rounded=0;whiteSpace=wrap;html=1;')
        ids.append(label_id)
    return ids, diagram


def draw_arcs(parent_node_id2children_node_ids: Dict[str, List[str]],
              diagram: drawio_diagram) -> drawio_diagram:
    '''Draws arcs between the parent node to the child node. 
    Returns the drawio diagram instance.'''
    for parent_node_id, children_node_ids in parent_node_id2children_node_ids.items():
        for child_node_id in children_node_ids:
            diagram.add_link(parent_node_id, child_node_id,
                             style='endArrow=classic;edgeStyle=orthogonalEdgeStyle;')
    return diagram


def child_dom_id(child_dom: SectionDOM | SubSecDOM | TextGroupDOM):
    '''If child is a TextGroup, only connect to the first text line'''
    if isinstance(child_dom, TextGroupDOM):
        return id(child_dom.text_lines[0])
    return id(child_dom)


def make_dom_tree(root_dom: SectionDOM) -> Tuple[List[str], List[int], List[List[int]]]:
    dom_stack = [root_dom]
    labels: List[str] = []
    dom_instance_ids: List[int] = []
    # What the DOM Instance Ids are pointed to
    children_dom_instance_ids: List[List[int]] = []

    while len(dom_stack):
        dom = dom_stack.pop()
        if isinstance(dom, DocumentDOM):
            # 1. Add the label of DOM to labels
            # 2. Add the id of the DOM to the dom_instance_ids
            # 3. Add the id of the children to the children_dom_instance_ids
            # 4. Add the children to dom_stack
            labels.append('DOCUMENT')
            dom_instance_ids.append(id(dom))
            children_dom_instance_ids.append(
                list(map(child_dom_id, dom.children)))
            dom_stack += dom.children
            continue
        if isinstance(dom, SectionDOM):
            labels.append(dom.section_title)
            dom_instance_ids.append(id(dom))

            children_dom_instance_ids.append(
                list(map(child_dom_id, dom.children)))
            dom_stack += dom.children
            continue
        if isinstance(dom, SubSecDOM):
            labels.append(dom.subsec_title)
            dom_instance_ids.append(id(dom))
            if dom.text_group is None:  # No child to append
                children_dom_instance_ids.append([])
                continue
            # Append the first string in text_lines as the child of the SubSecDom
            children_dom_instance_ids.append([child_dom_id(dom.text_group)])
            dom_stack.append(dom.text_group)
            continue
        if isinstance(dom, TextGroupDOM):
            labels += dom.text_lines
            dom_instance_ids += list(map(id, dom.text_lines))
            # All connected to the immediate next nodes
            children_dom_instance_ids += \
                [[id(dom.text_lines[i+1])]
                 for i in range(len(dom.text_lines)-1)]
            # The last node points to no child node
            children_dom_instance_ids += [[]]
            continue
        raise ValueError(
            f'Stack contained something other than DocumentDOM, SectionDOM, SubSecDOM or TextGroupDOM: {type(dom)}')

    # Check that labels, dom_instance_ids and children_dom_instance_ids all have the same length
    if len(labels) != len(dom_instance_ids) or len(dom_instance_ids) != len(children_dom_instance_ids):
        raise ValueError(
            f'The length of labels, dom_instance_ids and must be the same but were: {len(labels), len(dom_instance_ids), len(children_dom_instance_ids)}')
    return labels, dom_instance_ids, children_dom_instance_ids


def find_max_width(dom_instance_ids: List[int], dom_instance_id2children_dom_instance_ids: Dict[int, List[int]]) -> Dict[int, int]:
    '''Returns the maximum width of children stemming from it's own node. 
    Returns a dictionary mapping dom_instance_id to the sum of width of it's decendent DOMs.'''
    dom_instance_id2children_dom_instance_ids = copy.deepcopy(
        dom_instance_id2children_dom_instance_ids)  # Copy it so that it doesn't affect the input instance
    dom_instance_id2children_max_width: Dict[int, List[int]] = dict(
        [(dom_instance_id, []) for dom_instance_id in dom_instance_ids])
    parent_dom_stack: List[Optional[int]] = [None]
    current_dom_stack: List[int] = [dom_instance_ids[0]]

    while len(parent_dom_stack) > 0:
        parent_dom: Optional[int] = parent_dom_stack.pop()
        current_dom: int = current_dom_stack.pop()
        # DOM still has children DOM to add to queue
        if len(dom_instance_id2children_dom_instance_ids[current_dom]):
            # Add the current dom to stack
            parent_dom_stack.append(parent_dom)
            current_dom_stack.append(current_dom)
            # Then Add it's childrent dom to stack
            while len(dom_instance_id2children_dom_instance_ids[current_dom]) != 0:
                parent_dom_stack.append(current_dom)
                current_dom_stack.append(
                    dom_instance_id2children_dom_instance_ids[current_dom].pop())
            continue

        if parent_dom is None:
            break   # Reached the root node
        # Contains no children
        if len(dom_instance_id2children_max_width[current_dom]) == 0:
            # Leaf Node therefore set it's own width to 1
            dom_instance_id2children_max_width[current_dom].append(1)
        # Add it's own max width to it's parent
        dom_instance_id2children_max_width[parent_dom].append(
            sum(dom_instance_id2children_max_width[current_dom]))
    # Aggregate dom_instance_id2children_max_width
    dom_instance_id2max_decendent_width = {}
    for dom_instance_id, children_max_width in dom_instance_id2children_max_width.items():
        dom_instance_id2max_decendent_width[dom_instance_id] = sum(
            children_max_width)
    return dom_instance_id2max_decendent_width


def calculate_label_positions(dom_instance_ids: List[int],
                              dom_instance_id2children_dom_instance_ids: Dict[int, List[int]],
                              dom_instance_id2max_decendent_width: Dict[int, int]) -> List[Tuple[float, float]]:
    '''Calculates the position of each DOM.'''
    dom_instance_id2position: Dict[int, Tuple[float, float]] = {}
    dom_id_stack: List[int] = [dom_instance_ids[0]]
    x0_stack: List[int] = [0]
    y0_stack: List[int] = [0]
    while len(dom_id_stack) > 0:
        dom_id = dom_id_stack.pop()
        # x0 and y0 are where the rectangle should start
        # _____________
        # |<-x0         |
        # |-------------|
        # |      |      |
        # -------------
        x_zero = x0_stack.pop()
        y_zero = y0_stack.pop()
        x_pos = x_zero + (dom_instance_id2max_decendent_width[dom_id]/2) - 0.5
        y_pos = y_zero
        # Update current dom Position
        dom_instance_id2position[dom_id] = (x_pos, y_pos)

        # Add it's children to stack
        y_zero += 1     # Update y_zero for children
        for child in dom_instance_id2children_dom_instance_ids[dom_id]:
            dom_id_stack.append(child)
            x0_stack.append(x_zero)
            y0_stack.append(y_zero)
            # Update x_zero
            x_zero += dom_instance_id2max_decendent_width[child]
    return [dom_instance_id2position[dom_id] for dom_id in dom_instance_ids]


def substitute_children_dom_instance_ids_with_drawio_ids(children_dom_instance_ids: List[List[int]],
                                                         dom_instance_ids2drawio_ids: Dict[int, str]) -> List[List[str]]:
    children_drawio_ids: List[List[str]] = []
    for children_dom in children_dom_instance_ids:
        drawio_ids: List[str] = []
        for child_id in children_dom:
            drawio_ids.append(dom_instance_ids2drawio_ids[child_id])
        children_drawio_ids.append(drawio_ids)
    return children_drawio_ids


def draw_diagram(section_dom: SectionDOM, diagram: drawio_diagram) -> drawio_diagram:
    labels, dom_instance_ids, children_dom_instance_ids = make_dom_tree(
        section_dom)
    # Calculate the Position of each DOM
    dom_instance_ids2children_dom_instance_ids: Dict[int, List[int]] = dict(
        zip(dom_instance_ids, children_dom_instance_ids))
    dom_instance_id2max_decendent_width: Dict[int, int] = find_max_width(
        dom_instance_ids, dom_instance_ids2children_dom_instance_ids)
    dom_positions: List[Tuple[float, float]] = calculate_label_positions(
        dom_instance_ids, dom_instance_ids2children_dom_instance_ids, dom_instance_id2max_decendent_width)
    # Draw DOM
    drawio_id_generator = DrawioIdGenerator()
    drawio_ids, diagram = draw_nodes(
        labels, dom_positions, diagram, drawio_id_generator)
    dom_instance_ids2drawio_ids: Dict[int, str] = dict(
        zip(dom_instance_ids, drawio_ids))
    children_drawio_ids: List[List[str]] = substitute_children_dom_instance_ids_with_drawio_ids(
        children_dom_instance_ids, dom_instance_ids2drawio_ids)
    diagram = draw_arcs(dict(zip(drawio_ids, children_drawio_ids)), diagram)
    return diagram
