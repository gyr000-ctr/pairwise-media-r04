"""gridnoise 核心 API（初始未实现）。"""


class Terrain:
    def __init__(self, seed, octaves=4):
        """构造"""
        raise NotImplementedError

    def height(self, x, y):
        """高度"""
        raise NotImplementedError

    def tile_size(self):
        """平铺尺寸"""
        raise NotImplementedError
