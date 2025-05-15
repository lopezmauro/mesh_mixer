from maya import cmds
import maya.mel as mel
from maya.api import OpenMaya as om
import numpy as np
from mesh_mixer.core import maya_utils
NON_ASSIGNED_MASK_NAME = "non-assigned"

def create_vertex_color_mask(mesh_name, mask_name, default_color=(0, 0, 0, 1)):
    """
    Create a new vertex color set (mask) on the specified mesh
    
    Args:
        mesh_name (str): Name of the mesh to add the color set to
        mask_name (str): Name for the new vertex color set
        default_color (tuple): Default RGBA color values (0-1 range)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if mesh exists
    if not cmds.objExists(mesh_name):
        print(f"Error: Mesh '{mesh_name}' does not exist.")
        return False
    
    # Check if color set already exists
    existing_color_sets = cmds.polyColorSet(mesh_name, query=True, allColorSets=True) or []
    if mask_name in existing_color_sets:
        print(f"Warning: Color set '{mask_name}' already exists on '{mesh_name}'.")
        return False
    
    # Create the new color set
    cmds.polyColorSet(mesh_name, create=True, colorSet=mask_name, representation="RGBA")
    
    # Set the default color for all vertices
    cmds.select(mesh_name)
    #set_default_mask_color(mesh_name, mask_name, default_color)
    cmds.polyColorPerVertex(r=default_color[0], g=default_color[1], b=default_color[2], a=default_color[3], cdo=True)

    print(f"Created vertex color mask '{mask_name}' on '{mesh_name}'")
    return True

def activate_vertex_color_paint_tool(mesh_name, mask_name):
    """
    Activate the vertex color paint tool for a specific mask on the mesh
    
    Args:
        mesh_name (str): Name of the mesh
        mask_name (str): Name of the color set to paint on
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if mesh exists
    if not cmds.objExists(mesh_name):
        print(f"Error: Mesh '{mesh_name}' does not exist.")
        return False
    
    # Check if color set exists
    existing_color_sets = cmds.polyColorSet(mesh_name, query=True, allColorSets=True) or []
    if mask_name not in existing_color_sets:
        print(f"Error: Color set '{mask_name}' does not exist on '{mesh_name}'.")
        return False
    
    # Select the mesh
    cmds.select(mesh_name)
    
    # Set the current color set
    cmds.polyColorSet(mesh_name, currentColorSet=True, colorSet=mask_name)
    
    # Activate the Artisan tool with the Paint Vertex Color tool context
    mel.eval("PaintVertexColorTool")

    print(f"Activated vertex color paint tool for mask '{mask_name}' on '{mesh_name}'")
    return True

def list_vertex_color_masks(mesh_name):
    """
    List all vertex color sets (masks) on the specified mesh
    
    Args:
        mesh_name (str): Name of the mesh
    
    Returns:
        list: List of color set names
    """
    # Check if mesh exists
    if not cmds.objExists(mesh_name):
        print(f"Error: Mesh '{mesh_name}' does not exist.")
        return []
    
    # Get all color sets
    color_sets = cmds.polyColorSet(mesh_name, query=True, allColorSets=True) or []
    
    if not color_sets:
        print(f"No vertex color masks found on '{mesh_name}'.")
    
    return color_sets

def get_vertex_color_mask(mesh_name, mask_name):
    """
    Export vertex color mask data to a file for later retrieval
    
    Args:
        mesh_name (str): Name of the mesh
        mask_name (str): Name of the color set to export
    
    Returns:
        dict: Dictionary with the color data
    """
    # Check if mesh exists
    if not cmds.objExists(mesh_name):
        print(f"Error: Mesh '{mesh_name}' does not exist.")
        return None
    
    # Check if color set exists
    existing_color_sets = cmds.polyColorSet(mesh_name, query=True, allColorSets=True) or []
    if mask_name not in existing_color_sets:
        print(f"Error: Color set '{mask_name}' does not exist on '{mesh_name}'.")
        return None
    
    # Set the current color set
    cmds.polyColorSet(mesh_name, currentColorSet=True, colorSet=mask_name)
    
    # Get the number of vertices
    num_vertices = cmds.polyEvaluate(mesh_name, vertex=True)
    
    rgb = cmds.polyColorPerVertex(f'{mesh_name}.vtx[0:{num_vertices}]',  query=True, rgb=True)
    mask = np.max(np.asarray(rgb).reshape(-1, 3), axis=1)
    return mask

def set_vertex_color_mask(mesh_name, mask_name, mask_values):
    """
    Import vertex color mask data from a file or dictionary
    
    Args:
        file_path (str): Path to load the color data from (optional)
        color_data (dict): Dictionary with color data (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if mesh exists
    if not cmds.objExists(mesh_name):
        print(f"Error: Mesh '{mesh_name}' does not exist.")
        return False
    
    # Create color set if it doesn't exist
    existing_color_sets = cmds.polyColorSet(mesh_name, query=True, allColorSets=True) or []
    if mask_name not in existing_color_sets:
        cmds.polyColorSet(mesh_name, create=True, colorSet=mask_name, representation="RGBA")
    
    # Set the current color set
    cmds.polyColorSet(mesh_name, currentColorSet=True, colorSet=mask_name)
    

    # Create MColorArray with grayscale values
    colors = om.MColorArray()
    for val in mask_values:
        colors.append(om.MColor([val, val, val, 1.0]))  # RGBA

    # Create vertex indices
    indices = list(range(len(mask_values)))
    mesh_fn = maya_utils.get_mesh_fn(mesh_name)
    # Assign vertex colors
    mesh_fn.setVertexColors(colors, indices)
    


def delete_vertex_color_mask(mesh_name, mask_name):
    """
    Delete a vertex color set (mask) from the specified mesh
    
    Args:
        mesh_name (str): Name of the mesh
        mask_name (str): Name of the color set to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if mesh exists
    if not cmds.objExists(mesh_name):
        print(f"Error: Mesh '{mesh_name}' does not exist.")
        return False
    
    # Check if color set exists
    existing_color_sets = cmds.polyColorSet(mesh_name, query=True, allColorSets=True) or []
    if mask_name not in existing_color_sets:
        print(f"Error: Color set '{mask_name}' does not exist on '{mesh_name}'.")
        return False
    
    # Delete the color set
    cmds.polyColorSet(mesh_name, delete=True, colorSet=mask_name)
    
    print(f"Deleted vertex color mask '{mask_name}' from '{mesh_name}'")
    return True


def normalize_masks(masks_input):
    """
    Performs vectorized normalization on a dictionary of NumPy arrays representing weight layers for a 3D mesh.
    
    Args:
        masks_input (dict): Dictionary where keys are layer names (strings) and values are NumPy arrays
                             of shape (num_vertices) containing float weights.
    
    Returns:
        dict: New dictionary with the same keys but normalized weight values according to the specified rules.
    """
    # Create a copy of the input dictionary to avoid modifying the original
    normalized_layers = {}
    
    # Extract the non-assigned layer
    non_assigned_layer = masks_input.get(NON_ASSIGNED_MASK_NAME, None)
    if non_assigned_layer is None:
        masks_input[NON_ASSIGNED_MASK_NAME] = np.ones_like(masks_input[list(masks_input.keys())[0]])
    
    # Calculate the sum of weights for each vertex (excluding "non-assigned")
    sum_weights = np.zeros_like(non_assigned_layer)
    for layer_name, weights in masks_input.items():
        if layer_name != NON_ASSIGNED_MASK_NAME:
            sum_weights += weights
    
    # Create boolean masks for vertices with sum >= 1 and sum < 1
    sum_gte_one_mask = sum_weights >= 1.0
    sum_lt_one_mask = ~sum_gte_one_mask
    
    # Process each layer and apply normalization
    for layer_name, weights in masks_input.items():
        if layer_name == NON_ASSIGNED_MASK_NAME:
            # For "non-assigned" layer:
            # - Set to 0 where sum >= 1
            # - Set to (1 - sum) where sum < 1
            normalized_weights = np.zeros_like(weights)
            normalized_weights[sum_lt_one_mask] = 1.0 - sum_weights[sum_lt_one_mask]
        else:
            # Copy the original weights
            normalized_weights = weights.copy()
            
            # For vertices with sum >= 1, normalize weights
            if np.any(sum_gte_one_mask):
                # Get the normalization factor for vertices with sum >= 1 (to preserve relative scale)
                normalization_factor = sum_weights[sum_gte_one_mask]
                
                # Apply normalization only to vertices with sum >= 1
                normalized_weights[sum_gte_one_mask] = weights[sum_gte_one_mask] / normalization_factor
            
            # Vertices with sum < 1 keep their original weights (already copied)
        
        normalized_layers[layer_name] = normalized_weights
    
    return normalized_layers

def get_all_masks(mesh_name):
    """
    Get all masks from the specified mesh.

    Args:
        mesh_name (str): The name of the mesh (transform node).

    Returns:
        dict: A dictionary containing the mask names and their corresponding values.
    """
    # Get all masks
    masks = list_vertex_color_masks(mesh_name)
    regions_mask = dict()
    # Iterate over each mask attribute
    for mask in masks:
        regions_mask[mask]= get_vertex_color_mask(mesh_name, mask)
    
    return regions_mask