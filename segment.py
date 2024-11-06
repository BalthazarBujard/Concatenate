# file containing the segment class 

from typing import List
import numpy as np

class Segment():
    def __init__(self, data : np.ndarray, segment_idx : int, timestamp : List[int]):
        self.data = data
        self.segment_idx = segment_idx
        self.timestamp = timestamp