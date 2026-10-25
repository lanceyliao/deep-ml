import math
from typing import Dict, List, Tuple

def get_space_dim(space: dict) -> int:
    """Helper function to extract dimension from Gym space definition."""
    space_type = space.get("type")
    
    if space_type == "Box":
        # Product of all dimensions in the shape tuple
        return math.prod(space["shape"])
    elif space_type in ("Discrete", "MultiBinary"):
        return space["n"]
    else:
        raise ValueError(f"Unsupported space type: {space_type}")

def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.
    """
    # 1. Determine input and output dimensions
    input_dim = get_space_dim(obs_space)
    output_dim = get_space_dim(act_space)
    
    # 2. Construct the full sequence of layer dimensions: [in, h1, h2, ..., h_L, out]
    layer_dims = [input_dim] + list(hidden_layers) + [output_dim]
    
    layer_shapes = []
    total_params = 0
    
    # 3. Iterate through consecutive dimension pairs to build weight and bias shapes
    for in_feat, out_feat in zip(layer_dims[:-1], layer_dims[1:]):
        weight_shape = (in_feat, out_feat)
        bias_shape = (out_feat,)
        
        layer_shapes.append((weight_shape, bias_shape))
        
        # 4. Count parameters: weights + biases
        # (in_feat * out_feat) + out_feat
        total_params += (in_feat * out_feat) + out_feat
        
    return {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'layer_shapes': layer_shapes,
        'total_params': total_params
    }