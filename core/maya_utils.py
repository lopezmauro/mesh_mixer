from maya.api import OpenMaya as om
from maya import OpenMayaUI as omui
from shiboken2 import wrapInstance
from PySide2 import QtWidgets 

def maya_main_window():
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QWidget)

def get_node_dag_path(node_name):
    """
    Get the DAG path of a node in Maya.

    Parameters:
    node_name (str): The name of the node to query.

    Returns:
    MDagPath: The DAG path of the node.
    """
    selection_list = om.MSelectionList()
    selection_list.add(node_name)
    return selection_list.getDagPath(0)

def get_mesh_fn(object_name):
    """
    Get the MFnMesh object of the selected mesh in Maya.

    Returns:
    MFnMesh: The MFnMesh object of the selected mesh.
    """
    node_dag_path = get_node_dag_path(object_name)
    # Create an MFnMesh object
    fn_mesh = om.MFnMesh(node_dag_path)
    return fn_mesh

