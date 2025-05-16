# Mesh Mixer

Mesh Mixer is a Python-based tool for exploring machine learning techniques applied to 3D mesh manipulation. It allows users to compress meshes by region into PCA values, then mix and manipulate the latent space of those regions.

⚠️ **Disclaimer**: This software is provided for educational and research purposes only. It is not intended for production use. Use at your own risk.

---

## Features

- **Mesh Selection and Management**: Add, remove, and manage 3D meshes for processing.
- **Mask Creation**: Create and manage vertex color masks for specific regions of meshes.
- **Latent Space Mixing**: Explore and manipulate the latent space of 3D meshes using machine learning techniques.
- **Integration with Maya**: Designed to work seamlessly within Autodesk Maya's environment.


---
#### Mask creation
https://github.com/user-attachments/assets/fa38bc87-8e23-42c5-b947-5794d36b0fc2

#### Latent interpolation
https://github.com/user-attachments/assets/2fe4d957-e500-4bc3-bd09-97f946a7bbd0

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/mesh_mixer.git
   cd mesh_mixer
   ```

2. Install the required Python dependencies:
   - numpy
   - scipy
   - scikit-learn

3. Ensure Autodesk Maya is installed and properly configured to run Python scripts.

---

## Usage

### Launching the Main Window

1. Open Autodesk Maya.
2. Run the following script in Maya's Python script editor to launch the Mesh Mixer UI:

   ```python
   import sys
   # Add the path to the Mesh Mixer directory
   # Replace <path_to_mesh_mixer> with the absolute path to the Mesh Mixer repository on your system
   sys.path.append(r"<path_to_mesh_mixer>")
   from mesh_mixer.main_ui import MainWindow
   MainWindow().show()
   ```

3. Use the `MainWindow` to:
   - Add meshes to the mixer.
   - Create and manage masks for specific regions of the meshes.
   - Enable the "Open Mixer" button once meshes and masks are set up.

### Using the Latent Mixer

4. The `LatentMixer` UI will open, allowing you to:
   - Explore the latent space of the selected meshes.
   - Adjust parameters to mix and manipulate the meshes.
   - Visualize the results in real-time within Maya.

---

## File Structure

- **`core/`**: Contains the core logic for mesh manipulation and machine learning integration.
- **`ui/`**: Contains the user interface components for interacting with the tool.
- **`main_ui.py`**: The main entry point for launching the Mesh Mixer UI.
- **`README.md`**: Documentation for the repository.
- **`LICENSE`**: Licensing information.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
