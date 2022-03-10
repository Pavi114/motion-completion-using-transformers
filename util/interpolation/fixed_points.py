import imp
from torch import LongTensor

from constants import DEVICE

def get_fixed_points(window_size, keyframe_gap):
    fixed_points = list(range(0, window_size, keyframe_gap))

    if (window_size - 1) % keyframe_gap != 0:
        fixed_points.append(window_size - 1)

    fixed_points = LongTensor(fixed_points).to(DEVICE)

    return fixed_points