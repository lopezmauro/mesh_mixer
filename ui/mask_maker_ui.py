import maya.cmds as cmds
from PySide2 import QtCore, QtWidgets, QtGui
from mesh_mixer.core import maya_utils
from mesh_mixer.core import mask_utils
from mesh_mixer.core import pyside_utils

class MaskWidget(QtWidgets.QWidget):
    has_masks_Signal = QtCore.Signal(bool)
    def __init__(self, mixer, parent=None):
        # Get Maya's main window as parent
        super(MaskWidget, self).__init__(parent)
        # Create UI elements
        self.mixer = mixer
        self.create_ui()
        
        # Connect signals
        self.create_mask_btn.clicked.connect(self.create_new_mask)
        self.mask_list.itemClicked.connect(self.select_mask)
        self.refresh_btn.clicked.connect(self.refresh_mask_list)
        self.delete_btn.clicked.connect(self.delete_mask)
        
        # Initial refresh of the mask list
        self.refresh_mask_list()
    
    def create_ui(self):
        """Create the user interface elements"""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Create new mask button
        self.create_mask_btn = QtWidgets.QPushButton("Create New region Mask")
        self.create_mask_btn.setIcon(QtGui.QIcon(":/3dPaint.png"))  # Icon resembling a paintbrush
        main_layout.addWidget(self.create_mask_btn)
        
        # Separator line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        main_layout.addWidget(line)
        
        # Label for the list
        list_label = QtWidgets.QLabel("Available regions:")
        main_layout.addWidget(list_label)
        
        # List widget for masks
        self.mask_list = QtWidgets.QListWidget()
        main_layout.addWidget(self.mask_list)
        
        # Delete mask button
        self.delete_btn = QtWidgets.QPushButton("Delete regions mask")
        self.delete_btn.setToolTip("Delete the selected mask")
        self.delete_btn.setIcon(QtGui.QIcon(":/closeIcon.svg"))  # Icon resembling an "X"
        main_layout.addWidget(self.delete_btn)
        
        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: grey; font-style: italic;")
        main_layout.addWidget(self.status_label)
        
        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton("Refresh regions")
        self.refresh_btn.setIcon(QtGui.QIcon(":/refresh.png"))  # Maya "refresh" icon
        main_layout.addWidget(self.refresh_btn)

        # Normalize masks button
        self.normalize_masks_btn = QtWidgets.QPushButton("Normalize Regions")
        self.normalize_masks_btn.setToolTip("Normalize the weights of all masks")
        self.normalize_masks_btn.setIcon(QtGui.QIcon(":/normalize.png"))  # Icon resembling "normalize curves"
        self.normalize_masks_btn.clicked.connect(self.normalize_masks)
        main_layout.addWidget(self.normalize_masks_btn)
    
    @pyside_utils.wait_cursor
    def create_new_mask(self):
        """Create a new paintable attribute on the selected mesh"""
        # Check if a mesh is selected
        shape_node = self.mixer.get_mask_mesh()
        if not shape_node:
            self.status_label.setText("Error: No mask mesh created")
            return
        # Prompt for mask name
        result = cmds.promptDialog(
            title="New Mask",
            message="Enter mask name:",
            button=["OK", "Cancel"],
            defaultButton="OK",
            cancelButton="Cancel",
            dismissString="Cancel"
        )
        
        if result != "OK":
            return
        
        mask_name = cmds.promptDialog(query=True, text=True)
        if not mask_name:
            cmds.warning("Mask name cannot be empty.")
            self.status_label.setText("Error: Empty mask name")
            return
        # Make sure the name is valid and unique
        mask_name = self.validate_attribute_name(mask_name)
        
        try:
            mask_utils.create_vertex_color_mask(shape_node, mask_name, default_color=(0, 0, 0, 1))
            # Refresh the list
            self.refresh_mask_list()
            
            # Switch to Paint Attribute Tool
            mask_utils.activate_vertex_color_paint_tool(shape_node, mask_name)
            
            self.status_label.setText(f"Created mask: {mask_name}")
        except Exception as e:
            cmds.warning(f"Error creating mask: {str(e)}")
            self.status_label.setText("Error creating mask")

    def delete_mask(self):
        """Delete the selected mask from the mesh"""
        # Get the selected mask
        selected_items = self.mask_list.selectedItems()
        if not selected_items:
            cmds.warning("Please select a mask to delete.")
            self.status_label.setText("Error: No mask selected")
            return
        
        mask_name = selected_items[0].text()
        
        shape_node = self.mixer.get_mask_mesh()
        if not shape_node:
            self.status_label.setText("Error: No mask mesh created")
            return
        # Delete the mask
        try:
            mask_utils.delete_vertex_color_mask(shape_node, mask_name)
            self.refresh_mask_list()
            self.status_label.setText(f"Deleted mask: {mask_name}")
        except Exception as e:
            cmds.warning(f"Error deleting mask: {str(e)}")
            self.status_label.setText("Error deleting mask")


    def validate_attribute_name(self, name):
        """Ensure the attribute name is valid for Maya"""
        # Replace spaces with underscores
        name = name.replace(" ", "_")
        
        # Make sure it starts with a letter
        if not name[0].isalpha():
            name = "m_" + name
        
        return name
    
    def refresh_mask_list(self):
        """Refresh the list of available masks on the selected mesh"""
        # Clear the current list
        self.mask_list.clear()
        shape_node = self.mixer.get_mask_mesh()
        if not shape_node:
            self.status_label.setText("Error: No mask mesh created")
            return
        # Get all attributes on the shape node
        masks = mask_utils.list_vertex_color_masks(shape_node)
        # Add masks to the list
        if masks:
            for mask in masks:
                self.mask_list.addItem(mask)
            self.status_label.setText(f"{len(masks)} mask(s) found")
        else:
            self.status_label.setText("No paintable masks found")
        # Emit signal if there are masks
        self.has_masks_Signal.emit(self.mask_list.count())
    
    def select_mask(self, item):
        """Select a mask from the list and switch to the paint tool"""
        # Get the selected mask name
        mask_name = item.text()
        # Switch to Paint Attribute Tool for this mask
        shape_node = self.mixer.get_mask_mesh()
        mask_utils.activate_vertex_color_paint_tool(shape_node, mask_name)
        
        self.status_label.setText(f"Selected mask: {mask_name}")
    
    def normalize_masks(self):
        """Normalize the weights of the selected masks"""
           
        # Get the selected mesh
        shape_node = self.mixer.get_mask_mesh()
        if not shape_node:
            self.status_label.setText("Error: No mask mesh created")
            return
        masks = mask_utils.get_all_masks(shape_node)
        
        # Normalize the weights
        normalized_masks = mask_utils.normalize_masks(masks)
        for mask in normalized_masks:
            mask_utils.set_vertex_color_mask(shape_node, mask, normalized_masks[mask])
        self.refresh_mask_list()
        self.status_label.setText(f"All masks normalized")

class MaskMakerUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        if parent is None:
            parent = maya_utils.maya_main_window()
        super(MaskMakerUI, self).__init__(parent)
        self.setWindowTitle("Mask Maker")
        self.setMinimumWidth(300)
        self.setWindowFlags(QtCore.Qt.Window)
        
        # Create the mask widget
        self.mask_widget = MaskWidget(parent=self)
        
        # Set the layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.mask_widget)
        
        # Set the main layout
        self.setLayout(layout)
