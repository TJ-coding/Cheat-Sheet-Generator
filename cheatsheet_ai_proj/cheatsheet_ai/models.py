from typing import List, Tuple, NamedTuple

class Graph(NamedTuple): 
    ids: List[str]
    titles: List[str]
    xs: List[float]
    ys: List[float]
    widths: List[float]
    heights: List[float]
    coo_adjacency_matrix: List[Tuple[int, int]]