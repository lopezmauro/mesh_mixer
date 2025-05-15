from maya.api import OpenMaya as om
from maya import cmds
import numpy as np
from sklearn.decomposition import PCA
from mesh_mixer.core import mask_utils
from mesh_mixer.core import maya_utils

class MeshMixer:
    def __init__(self):
        self.all_points = None
        self.num_meshes = None
        self.num_points = None
        self.regions_mask = None
        self.regions_indices = None
        self.average_normalized = None
        self.current_mesh = None
        self.stored_mesh = None
        self.average_std = None
        self.average_mean = None
        self.normalized_vertices = None
        self.pca = None
        self.preserved_variance = 0.9
        self.meshes = None
        self.average_mesh = None
        self.average_mesh_fn = None
        self.mask_mesh = None

    def reset(self):
        self.all_points = None
        self.num_meshes = None
        self.num_points = None
        self.mask_mesh = None
        self.regions_mask = None
        self.regions_indices = None
        self.average_normalized = None
        self.current_mesh = None
        self.stored_mesh = None
        self.average_std = None
        self.average_mean = None
        self.normalized_vertices = None
        self.pca = None
        self.preserved_variance = 0.9
        self.meshes = None
        self.average_mesh = None
        self.average_mesh_fn = None
        self.regions_pca = None


    def set_meshes(self, meshes):
        self.reset()
        self.meshes = meshes

    def clear_meshes(self):
        self.reset()

    def get_meshes_data(self):
        if self.meshes is None:
            raise Exception("No meshes set, please call set_meshes()")
        self.all_points = get_all_points(self.meshes)
        self.num_meshes = len(self.meshes)
        self.num_points = self.all_points.shape[0]

    def get_region_masks(self):
        if self.meshes is None:
            raise Exception("No meshes set, please call set_meshes()")
        if self.mask_mesh is None:
            self.mask_mesh = self.get_mask_mesh()
        # Get all masks
        self.regions_mask = get_mesh_masks(self.mask_mesh)

    def get_region_stats(self):
        if self.all_points is None:
            raise Exception("Points not initialized, please call get_meshes_data()")
        if self.regions_mask is None:
            raise Exception("Regions mask not initialized, please call get_region_mask()")
        self.regions_indices = dict()
        for region, mask in self.regions_mask.items():
            self.regions_indices[region] = np.where(mask)[0]
        self.normalized_vertices, std, mean = normalize_points(self.all_points, 
                                                               self.regions_mask, 
                                                               self.regions_indices)
        self.average_normalized, self.average_std, self.average_mean = get_average_stats(self.normalized_vertices, 
                                                                                         std, 
                                                                                         mean)
        self.stored_mesh = self.average_normalized.copy()
        self.current_mesh = self.average_normalized.copy()

    def get_latent_spaces(self):
        self.regions_pca = dict()
        for region, indices in self.regions_indices.items():
            flatten_points = self.normalized_vertices[:, indices].reshape(self.num_meshes, -1)
            pca = PCA(n_components=self.preserved_variance)
            pca.fit(flatten_points)
            print(f"Explained variance ratio ({region}): {pca.explained_variance_ratio_}")
            points_transformed = pca.transform(flatten_points)
            self.regions_pca[region] = {'model': pca, 'pca_values': points_transformed}
    
    def reconstruct_from_latent(self, region, latent):
        model = self.regions_pca[region].get('model')
        mask = self.regions_mask[region]
        indices = self.regions_indices[region]
        points_inverse = model.inverse_transform(latent).reshape(-1, 3)
        dest_points = self.stored_mesh.copy()
        dest_points[indices] = points_inverse
        new_mesh = mix_meshes(self.current_mesh, dest_points, mask, weight=1)
        self.current_mesh = new_mesh

    def apply_deformation(self):
        denorm_points = denormalize_mesh(self.current_mesh, self.average_std, self.average_mean)
        if self.average_mesh is None:
            self.create_average_face()
        m_points = om.MPointArray(denorm_points)
        self.average_mesh_fn.setPoints(m_points)

    def create_average_face(self):
        self.average_mesh, self.average_mesh_fn = create_temp_mesh(self.meshes[0], self.average_normalized)
        denorm_points = denormalize_mesh(self.average_normalized, self.average_std, self.average_mean)
        m_points = om.MPointArray(denorm_points)
        self.average_mesh_fn.setPoints(m_points)
    
    def store_current_mesh(self):
        self.stored_mesh = self.current_mesh.copy()

    def create_mask_mesh(self):
        if self.meshes is None:
            return
        if self.all_points is None:
            self.get_meshes_data()
        cmds.hide(self.meshes)
        self.mask_mesh, mask_fn = create_temp_mesh(self.meshes[0], np.mean(self.all_points, axis=0))
        if self.regions_mask:
            for region, mask in self.regions_mask.items():
                mask_utils.set_vertex_color_mask(self.mask_mesh, region, mask)
        else:
            num_vertices = mask_fn.numVertices
            mask_values = np.ones(num_vertices, dtype=float)
            mask_utils.set_vertex_color_mask(self.mask_mesh, mask_utils.NON_ASSIGNED_MASK_NAME, mask_values)

    def hide_mask_mesh(self):
        if self.mask_mesh:
            cmds.hide(self.mask_mesh)

    def get_mask_mesh(self):
        if not self.mask_mesh or not cmds.objExists(self.mask_mesh):
            self.create_mask_mesh()
        return self.mask_mesh
    
    def set_mask_mesh(self, mesh_name):
        self.mask_mesh = None
        if cmds.objExists(mesh_name):
            self.mask_mesh = mesh_name


        


def get_mesh_points(mesh_name):
    mesh_fn = maya_utils.get_mesh_fn(mesh_name)
    points = np.asarray(mesh_fn.getPoints(om.MSpace.kWorld))[:, :3]
    return points

def get_mesh_masks(mesh_name):
    # Get the shape node
    masks = mask_utils.get_all_masks(mesh_name)

    normalized_masks = mask_utils.normalize_masks(masks)
    return normalized_masks

def get_all_points(all_heads):
    neutral_points = get_mesh_points(all_heads[0])
    all_points = np.zeros((len(all_heads), len(neutral_points), 3))
    all_points[0] = neutral_points
    for i, mesh_name in enumerate(all_heads[1:]):
        neutral_points = get_mesh_points(mesh_name)
        all_points[i + 1] = neutral_points
    return all_points

def normalize_points(points, regions_mask, regions_indices):
    std = np.zeros_like(points)
    mean = np.zeros_like(points)
    for region, indices in regions_indices.items():
        region_points = points[:, indices]
        region_mean = np.mean(region_points, axis=1)
        region_std = np.std(region_points, axis=1)
        mean += region_mean[:, None, :] * regions_mask[region][None, :, None]
        std += region_std[:, None, :] * regions_mask[region][None, :, None]
    normalized_vertices = (points - mean) / std
    return normalized_vertices, std, mean

def get_average_stats(normalized_vertices, std, mean):
    average_normalized = np.mean(normalized_vertices, axis=0)
    average_std = np.mean(std, axis=0)
    average_mean = np.mean(mean, axis=0)
    return average_normalized, average_std, average_mean

def denormalize_mesh(normalized_pointd, std, mean):
    return (normalized_pointd * std) + mean

def mix_meshes(src_points, dst_points, mask, weight=1.0):
    weighted_mask = np.multiply(mask, weight)
    new_points = (src_points * (1 - weighted_mask[:, None])) + (dst_points * weighted_mask[:, None])
    return new_points

def create_temp_mesh(mesh_name, points, name='temp_mesh'):
    dupli = cmds.duplicate(mesh_name, n=name)[0]
    cmds.showHidden(dupli)
    mesh_fn = maya_utils.get_mesh_fn(dupli)
    m_points = om.MPointArray(points)
    mesh_fn.setPoints(m_points)
    return dupli, mesh_fn