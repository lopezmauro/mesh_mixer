from maya import cmds
import maya.mel as mel
from maya.api import OpenMaya as om
import numpy as np
from mesh_mixer.core import maya_utils

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
    
    # Print the color sets
    if color_sets:
        print(f"Vertex color masks on '{mesh_name}':")
        for i, cs in enumerate(color_sets):
            print(f"  {i+1}. {cs}")
    else:
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

def normalize_masks(masks):
    """
    Normalize the values of the given mask attributes on the specified mesh.

    Args:
        mesh_name (str): The name of the mesh (transform node).
        mask_attributes (list): A list of mask attribute names to normalize.
    """
    num_vertices = len(list(masks.values())[0])
    if 'non-assigned' not in masks:
        masks['non-assigned'] = np.zeros(num_vertices)

    zero_vertices = np.where(np.sum(list(masks.values()), axis=0) < 1e-5)[0]
    if zero_vertices.size > 0:
        masks['non-assigned'] = np.zeros(num_vertices)
        masks['non-assigned'][zero_vertices] = 1

    mask_names = list(masks.keys())
    mask_values = list(masks.values())
    mask_array = np.asarray(mask_values)
    normalized_weights = np.zeros_like(mask_array, dtype=float)
    non_asiggend_index = mask_names.index('non-assigned')
    for i in range(mask_array.shape[1]):
        vertex_weights = mask_array[:, i]
        non_zero_indices = np.nonzero(vertex_weights)[0]
        num_influences_for_vertex = len(non_zero_indices)

        if num_influences_for_vertex > 1:
            relevant_weights = vertex_weights[non_zero_indices]
            sum_relevant_weights = np.sum(relevant_weights)
            if sum_relevant_weights < 1:
                normalized_weights[non_asiggend_index, i] = 1.0 - sum_relevant_weights
                normalized_weights[non_zero_indices, i] = relevant_weights
            else:
                normalized_relevant_weights = relevant_weights / sum_relevant_weights
                normalized_weights[non_zero_indices, i] = normalized_relevant_weights
        elif num_influences_for_vertex == 1:
            value = mask_array[non_zero_indices[0], i]
            if value >= 1 or non_zero_indices[0] == non_asiggend_index:
                # If the vertex has only one influence and it is the non-assigned mask, set it to 1.0
                normalized_weights[non_zero_indices[0], i] = 1.0
            else:# If the vertex has only one influence split the value between the non-assigned 
                normalized_weights[non_asiggend_index, i] = 1.0 - value
                normalized_weights[non_zero_indices[0], i] = value
    result = dict()
    for i, mask_attr in enumerate(mask_names):
        # Set the normalized values back to the attribute
        result[mask_attr] = normalized_weights[i]
    return result

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