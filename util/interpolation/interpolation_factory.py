from re import I

from util.interpolation.spherical_interpolation import spherical_interpolation
from .linear_interpolation import linear_interpolation

interpolations = {
    'linear': linear_interpolation,
    'spherical': spherical_interpolation
}

def get_p_interpolation(interpolation: str):
    return linear_interpolation

def get_q_interpolation(interpolation: str):
    return interpolations[interpolation]