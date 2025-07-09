import re
import numpy as np


class Region:
    def __init__(self, x, y, lvl, size_x, size_y) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.x = x
        self.y = y
        self.lvl = lvl

    def center(self, downsample):
        cx = self.x + downsample*self.size_x/2
        cy = self.y + downsample*self.size_y/2
        return [cx,cy]


    def __repr__(self) -> str:
        return f"({self.x},{self.y}), lvl={self.lvl}, size=({self.size_x},{self.size_y})"
    
    @classmethod
    def from_str(cls, s):
        xy_tuple = re.search('\((.*)\), lvl=', s)
        size_tuple = re.search(' size=\((.*)\)', s)
        lvl_str = re.search(' lvl=(.*), ', s)

        x,y = map(int,xy_tuple.group(1).split(','))
        size_x, size_y = map(int,size_tuple.group(1).split(','))
        lvl = int(lvl_str.group(1))
        return cls(x, y, lvl, size_x, size_y)

class RegionNonRect:
    def __init__(self, pointA, pointB, pointC, pointD, lvl, size_x, size_y) -> None:
        self.pointA = pointA
        self.pointB = pointB
        self.pointC = pointC
        self.pointD = pointD
        self.lvl = lvl
        self.size_x = size_x
        self.size_y = size_y
        self._assert_valid()

    def _assert_valid(self):
        assert(\
        # A is left of B
         self.pointA[0] < self.pointB[0] and \
        # D is left of C
         self.pointD[0] < self.pointC[0] and \
        # A is above D
         self.pointA[1] < self.pointD[1] and \
        # B is above C
         self.pointB[1] < self.pointC[1]
         ),f"Invalid Region was declared: {self.pointA} {self.pointB} {self.pointC} {self.pointD}"
    
    def center(self):
        return 0.25*np.array(self.pointA) + 0.25*np.array(self.pointB) + 0.25*np.array(self.pointC) + 0.25*np.array(self.pointD)
        
    
    def __repr__(self) -> str:
        return f"({self.pointA},{self.pointB},{self.pointC},{self.pointD}), lvl={self.lvl}, size=({self.size_x},{self.size_y})"
        
    @classmethod
    def from_str(cls, s):
        ABCD_tuples = re.search('\((.*)\), lvl=', s)
        ABCD_tuples = ABCD_tuples.group(1)
        ABCD_tuples = ABCD_tuples.split('],[')
        ABCD_tuples = [s.replace('[','').replace(']','').replace('.','').replace('  ',' ').strip() for s in ABCD_tuples]
        pointA, pointB, pointC, pointD = map(list, [map(int,t.split(' ')) for t in ABCD_tuples])

        size_tuple = re.search(' size=\((.*)\)', s)
        lvl_str = re.search(' lvl=(.*), ', s)

        size_x, size_y = map(int,size_tuple.group(1).split(','))
        lvl = int(lvl_str.group(1))
        return cls(pointA, pointB, pointC, pointD, lvl, size_x, size_y)