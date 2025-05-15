from PySide2 import QtWidgets, QtCore, QtGui
import maya.cmds as cmds
from mesh_mixer.ui import mask_maker_ui
from mesh_mixer.ui import mesh_mixer_ui
from mesh_mixer.core import mesh_mixer
from mesh_mixer.core import maya_utils
from mesh_mixer.core import pyside_utils
import importlib
importlib.reload(mask_maker_ui)
importlib.reload(mesh_mixer_ui)
importlib.reload(mesh_mixer)
importlib.reload(maya_utils)
class MeshesWidget(QtWidgets.QWidget):
    has_items_Signal = QtCore.Signal(bool)

    def __init__(self, mixer, parent=None):
        # Get Maya's main window as parent
        super(MeshesWidget, self).__init__(parent)
        # Create UI elements
        self.mixer = mixer
        self.create_ui()
        
        # Connect signals
        self.add_mesh.clicked.connect(self.add_selection)
        self.remove_btn.clicked.connect(self.remove_selection)
        self.mesh_list.itemClicked.connect(self.select_mesh)

    
    def create_ui(self):
        """Create the user interface elements"""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Label for the list
        list_label = QtWidgets.QLabel("Meshes to mix:")
        main_layout.addWidget(list_label)
        
        # List widget for masks
        self.mesh_list = QtWidgets.QListWidget()
        self.mesh_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        main_layout.addWidget(self.mesh_list)
        buttons_layout = QtWidgets.QHBoxLayout(self)
        self.add_mesh = QtWidgets.QPushButton("Add selected meshes")
        self.add_mesh.setToolTip("Add selected meshes to the list")
        self.add_mesh.setIcon(QtGui.QIcon(":/addClip.png"))  # Default Maya "add" icon
        buttons_layout.addWidget(self.add_mesh)
        self.remove_btn = QtWidgets.QPushButton("Remove selected meshes")
        self.remove_btn.setToolTip("Delete the selected mask")
        self.remove_btn.setIcon(QtGui.QIcon(":/deleteClip.png"))  # Default Maya "remove" icon
        buttons_layout.addWidget(self.remove_btn)
        main_layout.addLayout(buttons_layout)

    
    def add_selection(self):
        """Add selected meshes to the list"""
        selection = cmds.ls(selection=True)
        selected_meshes = [a for a in selection if cmds.objectType(a) == 'mesh']
        selected_transforms = [a for a in selection if cmds.objectType(a) == 'transform']
        shapes = cmds.listRelatives(selected_transforms, shapes=True, ni=True)
        selected_meshes.extend(shapes)
        if not selected_meshes:
            raise ValueError("No meshes selected")
        meshes = [cmds.listRelatives(mesh, parent=True)[0] for mesh in selected_meshes]
        meshes = list(set(meshes))
        for mesh in meshes:
            if not self.mesh_list.findItems(mesh, QtCore.Qt.MatchExactly):
                self.mesh_list.addItem(mesh)
        self.refresh_mixer()

    def refresh_mixer(self):
        """Refresh the mesh mixer with the current list of meshes"""
        num_meshes = self.mesh_list.count()
        self.has_items_Signal.emit(bool(num_meshes))
        if not num_meshes:
            self.mixer.clear_meshes()
            return
        all_meshes = [self.mesh_list.item(i).text() for i in range(self.mesh_list.count())]
        self.mixer.set_meshes(all_meshes)

    def remove_selection(self):
        """Remove selected meshes from the list"""
        selected_items = self.mesh_list.selectedItems()
        for item in selected_items:
            row = self.mesh_list.row(item)
            self.mesh_list.takeItem(row)
        self.refresh_mixer()

    def select_mesh(self):
        """Select the mesh in the viewport"""
        selected_items = self.mesh_list.selectedItems()
        if selected_items:
            selected_meshes = [item.text() for item in selected_items]
            cmds.select(selected_meshes)
        else:
            cmds.select(clear=True)

class MainWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        if parent is None:
            parent = maya_utils.maya_main_window()
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Mesh Mixer")
        self.setGeometry(300, 300, 600, 300)
        
        self.mixer = mesh_mixer.MeshMixer() # Initialize the mixer
        # Create the layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.splitter)
        
        # Create the meshes widget
        self.meshes_widget = MeshesWidget(self.mixer, self)
        self.splitter.addWidget(self.meshes_widget)
        
        self.mask_widget = mask_maker_ui.MaskWidget(self.mixer, self)
        self.mask_widget.setDisabled(True)
        self.splitter.addWidget(self.mask_widget)

        self.open_mix_button = QtWidgets.QPushButton("Open Mixer")
        self.open_mix_button.setDisabled(True)
        self.open_mix_button.setToolTip("Open the mesh mixer")
        self.open_mix_button.setIcon(QtGui.QIcon(":/out_character.png"))  # Default Maya "open" icon
        self.layout.addWidget(self.open_mix_button)

        self.meshes_widget.has_items_Signal.connect(self.mask_widget.setEnabled)
        self.mask_widget.has_masks_Signal.connect(self.open_mix_button.setEnabled)
        self.open_mix_button.clicked.connect(self.open_mixer)


    @pyside_utils.wait_cursor
    def open_mixer(self):
        if not self.mixer.meshes:
            raise ValueError("No meshes to mix")
        if not self.mixer.mask_mesh :
            raise ValueError("Plead add at least on region mask")
        self.mixer.get_meshes_data()
        self.mixer.get_region_masks()
        self.mixer.get_region_stats()
        self.mixer.get_latent_spaces()
        self.mixer.delete_mask_mesh()
        self.mixer.create_average_face()
        mixer_ui = mesh_mixer_ui.LatentMixer(self.mixer)
        mixer_ui.show()