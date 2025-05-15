import numpy as np
from PySide2 import QtWidgets, QtGui, QtCore
from sklearn import neighbors as sk_neighbors
from sklearn import cluster as sk_cluster
from sklearn import kernel_ridge as sk_kernel_ridge
from mesh_mixer.core import pyside_utils
from mesh_mixer.core import maya_utils
from mesh_mixer.ui import mask_maker_ui
PADDING = 0.2  # padding around the data points for better visualization

class LatentSpaceWidget(QtWidgets.QWidget):
    """PySide widget for latent space visualization with density heatmap"""
    point_clicked = QtCore.Signal(dict)

    def __init__(self, region, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.original_data = None  # Store original data for lookups
        self.data_2d = None        # 2D data for the representation
        self.representative_indices = None # Indices of the most representative
                                            # points
        self.clicked_point = None  # Pick the most representative indices
        self.selected_point = None # Where the nearest clicked point is
        self.show_density = True   # If we draw the density map
        self.density_map = None    # If created stored here to nor recreate it
                                    # every time
        self.x_range = None        # 2d data min and max values on x
        self.y_range = None        # 2d data min and max values on y
        self.title = f"{region} Latent Space Representation"

        self.point_size = 10
        self.point_color = QtGui.QColor(255, 255, 255)
        self.point_border_color = QtGui.QColor(0, 0, 0)

        self.colors = {
            'background': QtGui.QColor(255, 255, 255),
            'grid': QtGui.QColor(200, 200, 200),
            'clicked_point': QtGui.QColor(255, 0, 0),
            'selected_point': QtGui.QColor(200, 140, 0)
        }

        self.interpolator = None

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.setVisible(False)
        layout.addWidget(self.status_bar)


    def set_data(self, original_data):
        self.original_data = original_data
        self.data_2d = original_data[:, :2] # Just first two PCs for 2D representation
        pad = (self.data_2d.max(axis=0) - self.data_2d.min(axis=0)) * PADDING
        xmin, ymin = self.data_2d.min(axis=0) - pad
        xmax, ymax = self.data_2d.max(axis=0) + pad
        self.x_range = [xmin, xmax]
        self.y_range = [ymin, ymax]
        self.setup_interpolator()

    def set_point_appearance(self, size=None, color=None, border_color=None):
        if size is not None:
            self.point_size = size
        if color is not None:
            if isinstance(color, str):
                self.point_color = QtGui.QColor(color)
            elif isinstance(color, QtGui.QColor):
                self.point_color = color
        if border_color is not None:
            if isinstance(border_color, str):
                self.point_border_color = QtGui.QColor(border_color)
            elif isinstance(border_color, QtGui.QColor):
                self.point_border_color = border_color
        self.update()

    @pyside_utils.wait_cursor
    def get_density_map(self):
        if self.data_2d is None:
            return None
        if self.density_map is None:
            self.status_bar.setVisible(True)
            self.status_bar.showMessage("Generating Density map")
            self.density_map = create_heatmap_image_kde(self.data_2d,
                                                        image_size=(512, 512),
                                                        grid_range=[self.x_range[0], self.x_range[1],
                                                                    self.y_range[0], self.y_range[1]],
                                                        bandwidth=0.8,
                                                        grid_resolution=1.0,
                                                        color_deepth=100)
            self.status_bar.setVisible(False)
        return self.density_map

    def paintEvent(self, event):
        if self.data_2d is None:
            return
        

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Fill the background
        painter.fillRect(self.rect(), self.colors['background'])

        width = self.width()
        height = self.height()
        inner_rect = QtCore.QRectF(0, 0, width, height)
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        def data_to_screen(x, y):
            sx = (x - xmin) / (xmax - xmin) * width
            sy = height - (y - ymin) / (ymax - ymin) * height
            return sx, sy


        # Draw heatmap if available
        if self.show_density:
            density_map = self.get_density_map()
            if density_map is not None:
                density_map =  density_map.mirrored(False, True) # Flip the image vertically
                painter.drawImage(inner_rect, density_map)

        # Draw the scatter points
        points_to_plot = self.data_2d
        if self.representative_indices is not None:
            points_to_plot = self.data_2d[self.representative_indices]

        painter.setPen(QtGui.QPen(self.point_border_color, 1))
        radius = self.point_size // 2
        for i, point in enumerate(points_to_plot):
            if i == self.selected_point:
                painter.setBrush(QtGui.QBrush(self.colors['selected_point']))
            else:
                painter.setBrush(QtGui.QBrush(self.point_color))
            sx, sy = data_to_screen(point[0], point[1])
            painter.drawEllipse(int(sx - radius), int(sy - radius),
                                self.point_size, self.point_size)

        # Draw the clicked point if any
        if self.clicked_point is not None:
            painter.setPen(QtGui.QPen(self.colors['clicked_point'], 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            sx, sy = data_to_screen(self.clicked_point[0], self.clicked_point[1])
            line_size = 7
            painter.drawLine(int(sx - line_size), int(sy - line_size),
                            int(sx + line_size), int(sy + line_size))
            painter.drawLine(int(sx - line_size), int(sy + line_size),
                            int(sx + line_size), int(sy - line_size))
            

        # Draw axis labels
        painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        font = painter.font()
        font.setPointSize(15)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(self.title)
        title_rect = QtCore.QRectF(self.width() / 2 - text_rect.width() / 2, - 5,
                                  text_rect.width(), text_rect.height())
        painter.drawText(title_rect, QtCore.Qt.AlignCenter, self.title)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            width = self.width()
            height = self.height()
            x_scale = (self.x_range[1] - self.x_range[0]) / width if width > 0 else 1
            y_scale = (self.y_range[1] - self.y_range[0]) / height if height > 0 else 1
            
            if (0 <= event.x() <= width) and (0 <= event.y() <= height):

                x = self.x_range[0] + event.x() * x_scale
                y = self.y_range[1] - event.y() * y_scale # Invert y

                self.clicked_point = (x, y)
                interpolated_values = self.interpolate_at_point(x, y)
                nearest_idx, nearest_dist = self.get_nearest_point(x, y)
                
                if nearest_idx is not None:
                    self.selected_point = nearest_idx
                    nearest_point_coords = self.data_2d[nearest_idx]
                    nearest_original = self.original_data[nearest_idx]
                click_data = {
                    'coordinates': (x, y),
                    'interpolated_values': interpolated_values,
                    'nearest_point_idx': nearest_idx,
                    'nearest_point_distance': nearest_dist,
                    'nearest_point_coords': nearest_point_coords,
                    'original_data': nearest_original
                }
                self.point_clicked.emit(click_data)
                self.update()

    def setup_interpolator(self, gamma=0.8, alpha=0.001):
        if self.data_2d is None or self.original_data is None:
            return

        n_dims = self.original_data.shape[1]
        self.interpolator = []
        for i in range(n_dims):
            model = sk_kernel_ridge.KernelRidge(kernel='rbf', gamma=gamma, alpha=alpha)
            model.fit(self.data_2d, self.original_data[:, i])
            self.interpolator.append(model)

    def interpolate_at_point(self, x, y):
        if self.interpolator is None or len(self.interpolator) == 0:
            return None
        query_point = np.array([[x, y]])
        interpolated_values = np.zeros(len(self.interpolator))
        for i, model in enumerate(self.interpolator):
            interpolated_values[i] = model.predict(query_point)[0]
        return interpolated_values

    def get_nearest_point(self, x, y):
        if self.data_2d is None:
            return None, None
        query_point = np.array([[x, y]])
        distances = np.sqrt(np.sum((self.data_2d - query_point) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]
        return nearest_idx, min_distance

    def get_representative_points(self, percentage):
        """Set the most representative subset of points for visualization"""
        if self.data_2d is None:
            raise ValueError("No reduced vectors available.")
        if percentage >= 99:
            self.representative_indices = None
            self.update()
            return

        unique_points = np.unique(self.data_2d, axis=0)
        num_points = len(unique_points)
        num_representatives = max(1, int(num_points * percentage / 100))

        if num_representatives >= num_points:
            self.representative_indices = None
            self.update()
            return

        kmeans = sk_cluster.KMeans(n_clusters=num_representatives, random_state=42, n_init=10)
        kmeans.fit(unique_points)
        cluster_centers = kmeans.cluster_centers_
        self.representative_indices = []
        for center in cluster_centers:
            distances = np.sum((unique_points - center) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            self.representative_indices.append(closest_idx)
        self.update()

class LatentMixer(QtWidgets.QDialog):
    def __init__(self, mesh_mixer):
        # Get Maya's main window as parent
        maya_main_window = maya_utils.maya_main_window()
        super().__init__(maya_main_window)
        self.mixer = mesh_mixer
        self.latent_widgets = dict()  # placeholder for all latent widgets
        self.current_latent_widget = None  # placeholder for currently visible latent plot
        self.point_color = QtGui.QColor(255, 255, 255)
        self.point_border_color = QtGui.QColor(0, 0, 0)
        self.setWindowTitle("Latent Space Viewer ")
        self.setGeometry(100, 100, 1000, 700)

        # Create central widget
        #central_widget = QtWidgets.QWidget()
        #self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QtWidgets.QHBoxLayout(self)

        # create region selector
        #self.region_combo = QtWidgets.QComboBox()
        #self.region_combo.currentIndexChanged.connect(self.refresh_plot)
        #main_layout.addWidget(self.region_combo)
        main_mask_widget = QtWidgets.QWidget()
        mask_layout = QtWidgets.QVBoxLayout(main_mask_widget)
        main_layout.addWidget(main_mask_widget)
        
        # Label for the list
        list_label = QtWidgets.QLabel("Available Regions:")
        mask_layout.addWidget(list_label)
        
        # List widget for masks
        self.region_list = QtWidgets.QListWidget()
        self.region_list.itemClicked.connect(self.refresh_plot)
        mask_layout.addWidget(self.region_list)

        main_latent_widget = QtWidgets.QWidget()
        main_latent_layout = QtWidgets.QVBoxLayout(main_latent_widget)
        main_layout.addWidget(main_latent_widget, 1)
        # main widget that will hold all latent widgets
        latent_widget = QtWidgets.QWidget()
        self.latent_layout = QtWidgets.QVBoxLayout(latent_widget)
        main_latent_layout.addWidget(latent_widget, 1)

        settings_grp = QtWidgets.QGroupBox("Settings ")
        settings_layout = QtWidgets.QHBoxLayout(settings_grp)
        main_latent_layout.addWidget(settings_grp)

        self.density_cbx = QtWidgets.QCheckBox("Show Density ")
        self.density_cbx.setChecked(True)
        self.density_cbx.stateChanged.connect(self.refresh_plot)
        settings_layout.addWidget(self.density_cbx)
        # Representative points percentage slider
        rep_points_layout = QtWidgets.QHBoxLayout()
        rep_points_layout.addWidget(QtWidgets.QLabel("Representative Points : "))
        self.rep_points_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rep_points_slider.setRange(1, 100)
        self.rep_points_slider.setValue(100)  # Default to 100%
        self.rep_points_slider.setTickInterval(10)
        self.rep_points_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.rep_points_slider.valueChanged.connect(self.update_points_percentage)
        rep_points_layout.addWidget(self.rep_points_slider)
        self.rep_points_label = QtWidgets.QLabel("100 % ")
        rep_points_layout.addWidget(self.rep_points_label)
        settings_layout.addLayout(rep_points_layout)

        # Create controls container
        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_container)

        # Point appearance controls
        appearance_group = QtWidgets.QGroupBox("Point Appearance ")
        appearance_layout = QtWidgets.QFormLayout(appearance_group)

        # Point size control
        self.point_size_spinner = QtWidgets.QSpinBox()
        self.point_size_spinner.setRange(2, 30)
        self.point_size_spinner.setValue(10)
        self.point_size_spinner.valueChanged.connect(self.update_point_size)
        appearance_layout.addRow("Point Size : ", self.point_size_spinner)

        # Point color button
        self.point_color_button = QtWidgets.QPushButton("Set Point Color ")
        self.point_color_button.clicked.connect(self.choose_point_color)
        appearance_layout.addRow("Point Color : ", self.point_color_button)

        # Point border color button
        self.border_color_button = QtWidgets.QPushButton("Set Border Color ")
        self.border_color_button.clicked.connect(self.choose_border_color)
        appearance_layout.addRow("Border Color : ", self.border_color_button)
        controls_layout.addWidget(appearance_group)

        # Interpolation controls
        interp_group = QtWidgets.QGroupBox("Interpolation Settings ")
        interp_layout = QtWidgets.QFormLayout(interp_group)

        # Gamma slider ( for RBF kernel )
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setMinimum(1)  # 0.01
        self.gamma_slider.setMaximum(100)  # 1.0
        self.gamma_slider.setValue(8)  # 0.8
        self.gamma_slider.valueChanged.connect(self.setup_interpolation_models)
        self.gamma_label = QtWidgets.QLabel("Gamma : 0.10 ")
        interp_layout.addRow(self.gamma_label, self.gamma_slider)
        # Alpha slider ( regularization )
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setMinimum(1)  # 0.001
        self.alpha_slider.setMaximum(1000)  # 1.0
        self.alpha_slider.setValue(1)  # 0.001
        self.alpha_slider.valueChanged.connect(self.setup_interpolation_models)
        self.alpha_label = QtWidgets.QLabel("Alpha : 0.10 ")
        interp_layout.addRow(self.alpha_label, self.alpha_slider)
        controls_layout.addWidget(interp_group)

        # Info panel
        info_group = QtWidgets.QGroupBox("Point Information ")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        self.coords_label = QtWidgets.QLabel("Coordinates : - ")
        self.nearest_point_label = QtWidgets.QLabel("Nearest Point : - ")
        self.distance_label = QtWidgets.QLabel("Distance : - ")
        self.interpolated_label = QtWidgets.QLabel("Interpolated Data : - ")
        self.original_data_label = QtWidgets.QLabel("Original Data : - ")

        info_layout.addWidget(self.coords_label)
        info_layout.addWidget(self.nearest_point_label)
        info_layout.addWidget(self.distance_label)
        info_layout.addWidget(self.interpolated_label)
        info_layout.addWidget(self.original_data_label)

        controls_layout.addWidget(info_group)
        main_latent_layout.addWidget(controls_container)

        if self.mixer:
            self.populate_latent_widgets()
        self.refresh_plot()
        self.setup_interpolation_models()
    def open_mask_maker(self):
        """Open the mask maker UI"""
        self.mask_maker = mask_maker_ui.MaskMakerUI()
        self.mask_maker.show()
        self.mask_maker.finished.connect(self.populate_latent_widgets)
        self.mask_maker.refresh_mask_list()

    def populate_latent_widgets(self):
        #self.region_combo.clear()
        self.region_list.clear()
        for region in self.mixer.regions_pca.keys():
            self.region_list.addItem(region)
        #self.region_combo.addItems(self.mixer.regions_pca.keys())
        # Create latent space widgets and store latent space representations
        self.latent_widgets = dict()
        for region, data in self.mixer.regions_pca.items():
            pca_values = data.get('pca_values')
            widget = LatentSpaceWidget(region)
            widget.set_data(pca_values)
            self.latent_layout.addWidget(widget)
            widget.setVisible(False)
            widget.point_clicked.connect(self.on_point_clicked)
            self.latent_widgets[region] = widget

    def refresh_plot(self):
        # Generate some random data
        #region = self.region_combo.currentText()
        
        selected_items = self.region_list.selectedItems()
        if not selected_items:
            return
        region = selected_items[0].text()
        show_density = self.density_cbx.isChecked()
        for reg, widget in self.latent_widgets.items():
            widget.setVisible(False)
        self.current_latent_widget = self.latent_widgets.get(region)
        if self.current_latent_widget:
            self.current_latent_widget.show_density = show_density
            self.current_latent_widget.setVisible(True)
            self.set_current_widget_settings()
        self.mixer.store_current_mesh()

    def set_current_widget_settings(self):
        size = self.point_size_spinner.value()
        self.current_latent_widget.set_point_appearance(size=size)
        self.current_latent_widget.set_point_appearance(color=self.point_color, border_color=self.point_border_color)
        percentage = self.rep_points_slider.value()
        if percentage != 100:
            self.current_latent_widget.get_representative_points(percentage)

    def setup_interpolation_models(self):
        """Set up interpolation models with current settings"""
        # Get gamma and alpha from sliders
        gamma = self.gamma_slider.value() / 100.0
        alpha = self.alpha_slider.value() / 1000.0

        # Update Labels
        self.gamma_label.setText(f"Gamma : {gamma:.2f} ")
        self.alpha_label.setText(f"Alpha : {alpha:.3f} ")

        for widget in self.latent_widgets.values():
            widget.setup_interpolator(gamma=gamma, alpha=alpha)

    def update_point_size(self, size):
        """Update the point size in the visualization"""
        self.current_latent_widget.set_point_appearance(size=size)

    def choose_point_color(self):
        """Open color dialog to choose point color"""
        color = QtWidgets.QColorDialog.getColor(self.current_latent_widget.point_color, self, " Select Point Color ")
        if not color.isValid():
            return

        self.point_color = color
        self.current_latent_widget.set_point_appearance(color=color)

    def choose_border_color(self):
        """Open color dialog to choose point border color"""
        color = QtWidgets.QColorDialog.getColor(self.current_latent_widget.point_border_color, self, " Select Border Color ")
        if not color.isValid():
            return

        self.point_border_color = color
        self.current_latent_widget.set_point_appearance(border_color=color)

    def update_points_percentage(self, value):
        self.current_latent_widget.get_representative_points(value)

    def values_to_string(self, values):
        # Truncate if too long
        if len(values) > 5:
            return f"[{','.join(f'{x:.2f}' for x in values[:5])}...] "
        else:
            return f"[{','.join(f'{x:.2f}' for x in values)}] "

    def on_point_clicked(self, data):
        """Handle point click events"""
        x, y = data['coordinates']
        nearest_point = data['nearest_point_coords']
        distance = data['nearest_point_distance']
        interpolated = data['interpolated_values']
        original_data = data['original_data']

        # Update info labels
        self.coords_label.setText(f"Coordinates : ( {x:.2f} , {y:.2f} ) ")

        # Display interpolated values and reconstruct region
        if interpolated is not None:
            str_value = self.values_to_string(interpolated)
            self.interpolated_label.setText(f"Interpolated Data : {str_value} ")
            self.reconstruct_region_from_latent(interpolated)
        else:
            self.interpolated_label.setText("Interpolated Data : - ")

        # Display nearest point info
        if nearest_point is not None:
            self.nearest_point_label.setText(
                f"Nearest Point : ( {nearest_point[0]:.2f} , {nearest_point[1]:.2f} ) ")
            self.distance_label.setText(f"Distance : {distance:.4f} ")
        else:
            self.nearest_point_label.setText("Nearest Point : - ")
            self.distance_label.setText("Distance : - ")

        # Display original data if available
        if original_data is not None:
            str_value = self.values_to_string(original_data)
            self.original_data_label.setText(f"Original Data : {str_value} ")
        else:
            self.original_data_label.setText("Original Data : - ")

    def reconstruct_region_from_latent(self, latent_values):
        #region = self.region_combo.currentText()
        selected_items = self.region_list.selectedItems()
        if not selected_items:
            return
        region = selected_items[0].text()
        self.mixer.reconstruct_from_latent(region, latent_values)
        self.mixer.apply_deformation()

def bilinear_resize(img, new_shape):
    """
    Resize a 2-D (grayscale) or 3-D (color) image using bilinear interpolation.
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    # Scale factors
    row_ratio = (h - 1) / (new_h - 1) if new_h > 1 else 0
    col_ratio = (w - 1) / (new_w - 1) if new_w > 1 else 0

    # Coordinate grids of the target
    row_idx = np.arange(new_h)
    col_idx = np.arange(new_w)
    row_pos = row_idx * row_ratio  # float row positions in source
    col_pos = col_idx * col_ratio  # float col positions in source

    r0 = np.floor(row_pos).astype(int)  # top/left integer coordinates
    c0 = np.floor(col_pos).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)  # bottom/right integer coordinates
    c1 = np.minimum(c0 + 1, w - 1)

    # Fractional parts
    wr = row_pos - r0  # weight along rows
    wc = col_pos - c0  # weight along cols

    if img.ndim == 2:  # grayscale
        tl = img[r0[:, None], c0]  # top-left
        tr = img[r0[:, None], c1]  # top-right
        bl = img[r1[:, None], c0]  # bottom-left
        br = img[r1[:, None], c1]  # bottom-right
        top = tl * (1 - wc) + tr * wc  # top row
        bottom = bl * (1 - wc) + br * wc  # bottom row
        out = top * (1 - wr[:, None]) + bottom * wr[:, None]  # final output
        
    else:  # color - broadcast channel dim
        tl = img[r0[:, None], c0, :]
        tr = img[r0[:, None], c1, :]
        bl = img[r1[:, None], c0, :]
        br = img[r1[:, None], c1, :]
        top = tl * (1 - wc[:, None])[..., None] + tr * wc[:, None][..., None]  # Fixed broadcasting
        bottom = bl * (1 - wc[:, None])[..., None] + br * wc[:, None][..., None]  # Fixed broadcasting
        out = top * (1 - wr[:, None, None]) + bottom * wr[:, None, None]

    return out.astype(img.dtype)


def create_heatmap_image_kde(points,
                             image_size,
                             grid_range=None,
                             bandwidth=0.5,
                             grid_resolution=0.25,
                             color_deepth=100):
    """
    Generates a smooth heatmap image from 2D points using scikit-learn's
    Kernel Density Estimation (KDE) and PySide2.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) containing the 2D points.
        image_size (tuple): The desired output image dimensions (width, height).
        grid_range (list): Optional [xmin, xmax, ymin, ymax] for the grid.
        bandwidth (float): The bandwidth of the Gaussian kernel for KDE. Controls smoothness.
        grid_resolution (float): The resolution multiplier for the density grid.
        color_deepth (int): Number of colors in the color lookup table.
    """
    if points.shape[0] == 0:
        print("Warning: No points provided. Cannot generate heatmap.")
        img = QtGui.QImage(image_size[0], image_size[1], QtGui.QImage.Format_RGB32)
        img.fill(QtCore.Qt.black)
        return img

    # 1. Define Grid for KDE Evaluation
    if grid_range is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        grid_range = [xmin, xmax, ymin, ymax]
    
    # Create a grid of points where density will be evaluated
    # Make sure grid dimensions are integers
    grid_size = (int(image_size[0] * grid_resolution), int(image_size[1] * grid_resolution))
    
    # Create a grid of points where density will be evaluated
    xx, yy = np.meshgrid(np.linspace(grid_range[0], grid_range[1], grid_size[0]),
                         np.linspace(grid_range[2], grid_range[3], grid_size[1]))

    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T  # Shape (grid_size*grid_size, 2)

    # 2. Fit Kernel Density Estimator
    print(f"Fitting Kernel Density Estimator (bandwidth={bandwidth})...")
    # Use Gaussian kernel, adjust bandwidth as needed
    kde = sk_neighbors.KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(points)

    # 3. Evaluate Density on the Grid
    print("Evaluating density on grid...")
    # score_samples returns the log of the probability density
    log_density = kde.score_samples(grid_points)
    # Exponentiate to get actual density
    density = np.exp(log_density)
    # Reshape density back into a 2D grid
    density_grid = density.reshape(xx.shape)  # Use xx.shape to ensure correct dimensions

    # 4. Normalize Density for Coloring
    min_density = np.min(density_grid)
    max_density = np.max(density_grid)
    density_range = max_density - min_density
    if density_range <= 1e-9:
        density_range = 1  # Avoid division by zero

    normalized_density = (density_grid - min_density) / density_range

    # 5. create a color Look up table
    color_map = {
        0.0: (0.1, 0.002, 0.1),  # Dark Purple
        0.1: (0.127, 0.005, 0.329),  # Purple
        0.25: (0.300, 0.504, 0.550),  # Blue
        0.5: (0.700, 0.800, 0.400),  # Green
        0.8: (0.993, 0.900, 0.100),  # Yellow
        1.0: (0.906, 0.400, 0.100)  # Orange
    }
    key_pos = np.asarray(list(color_map.keys()))
    color_values = np.asarray(list(color_map.values()))
    target_pos = np.linspace(0.0, 1.0, color_deepth)
    lut = np.zeros((color_deepth, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(target_pos, key_pos, color_values[:, ch]) * 255

    # get the density as ints
    indices = (normalized_density * (len(lut) - 1)).astype(np.uint8)
    colorized_density = np.zeros((*normalized_density.shape, 3), dtype=np.uint8)
    
    # Apply the color map
    for i in range(3):  # For each color channel
        colorized_density[:, :, i] = lut[indices, i]

    # 6. Create QImage and Map Density to Pixels
    print("Generating image...")
    resized_grid = bilinear_resize(colorized_density, image_size)
    
    # Add alpha channel
    alpha_channel = np.full((image_size[0], image_size[1], 1), 255, dtype=np.uint8)
    rgba = np.concatenate((resized_grid, alpha_channel), axis=-1)  # shape (h, w, 4)
    image = QtGui.QImage(rgba.data, 
                        image_size[0], 
                        image_size[1], 
                        QtGui.QImage.Format_RGBA8888)
    
    # Create a copy to ensure the image data persists after the function returns
    return image.copy()

class MeshMixer:
    def __init__(self, mesh):
        self.mesh = mesh
        self.regions_pca = dict()  # Placeholder for PCA data
        self.current_region = None  # Placeholder for the current region
        self.current_mesh = None  # Placeholder for the current mesh

    def store_current_mesh(self):
        """Store the current mesh"""
        self.current_mesh = self.mesh

    def reconstruct_from_latent(self, region, latent_values):
        """Reconstruct the mesh from latent values"""
        # Placeholder for reconstruction logic
        pass

    def apply_deformation(self):
        """Apply deformation to the mesh"""
        # Placeholder for deformation logic
        pass