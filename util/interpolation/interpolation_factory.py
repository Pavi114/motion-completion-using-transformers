from .linear_interpolation import linear_interpolation

interpolations = {
    'linear': linear_interpolation
}

def get_interpolation(interpolation: str):
    return interpolations[interpolation]