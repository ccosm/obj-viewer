import slangpy as spy
import numpy as np
import trimesh
from PIL.Image import Image

class Mesh:
    def __init__(self, mesh_path: str, device: spy.Device):
        mesh = trimesh.load_mesh(mesh_path)
        positions = mesh.vertices.astype("float32")
        normals = mesh.vertex_normals.astype("float32")
        texcoords = mesh.visual.uv.astype("float32")  # type: ignore
        indices = mesh.faces.astype("uint16")
        self.index_format = spy.IndexFormat.uint16
        self.vertex_count = indices.size

        self.position_buffer = device.create_buffer(
            size=positions.nbytes,
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            data=positions,
        )

        self.normal_buffer = device.create_buffer(
            size=normals.nbytes,
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            data=normals,
        )

        self.uv_buffer = device.create_buffer(
            size=texcoords.nbytes,
            usage=spy.BufferUsage.vertex_buffer | spy.BufferUsage.shader_resource,
            data=texcoords,
        )

        self.index_buffer = device.create_buffer(
            size=indices.nbytes,
            usage=spy.BufferUsage.index_buffer | spy.BufferUsage.shader_resource,
            data=indices,
        )

        loader = spy.TextureLoader(device)
        image: Image = mesh.visual.material.image  # type: ignore
        image_shape = list(image.size) + [4]

        image_data = (
            np.frombuffer(image.tobytes(), dtype=np.uint8)
            .reshape(image_shape)
            .astype(np.float32)
            / 255
        )

        self.texture = loader.load_texture(spy.Bitmap(image_data))