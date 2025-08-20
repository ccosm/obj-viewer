import slangpy as spy
import numpy as np
import time
from pyglm import glm
from Mesh import Mesh

class MeshRenderer:
    def __init__(self, device: spy.Device, output_format: spy.Format):
        self.program = device.load_program(
            "phong.slang",
            ["vertex_main", "fragment_main"],
            link_options={"debug_info": spy.SlangDebugInfoLevel.maximal},
        )

        self.sampler = device.create_sampler()
        self.timer = time.perf_counter()

        self.pipeline = device.create_render_pipeline(
            program=self.program,
            targets=[{"format": output_format}],
            input_layout=device.create_input_layout(
                input_elements=[
                    {
                        "format": spy.Format.rgb32_float,
                        "semantic_name": "POSITION",
                        "buffer_slot_index": 0,
                    },
                    {
                        "format": spy.Format.rgb32_float,
                        "semantic_name": "NORMAL",
                        "buffer_slot_index": 1,
                    },
                    {
                        "format": spy.Format.rg32_float,
                        "semantic_name": "TEXCOORD",
                        "buffer_slot_index": 2,
                    },
                ],
                vertex_streams=[{"stride": 12}, {"stride": 12}, {"stride": 8}],
            ),
            depth_stencil={
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
                "format": spy.Format.d32_float
            },
        )

    def render(
        self,
        command_encoder: spy.CommandEncoder,
        mesh: Mesh,
        window_size: tuple[int, int],
        output_texture: spy.Texture,
        depth_texture: spy.Texture,
    ):
        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": output_texture.create_view(),
                        "clear_value": [0.1, 0.2, 0.3, 1.0],
                    }
                ],
                "depth_stencil_attachment": {
                    "view": depth_texture.create_view(),
                },
            }
        ) as pass_encoder:
            shader_object = pass_encoder.bind_pipeline(self.pipeline)
            cursor = spy.ShaderCursor(shader_object)
            cursor.sampler = self.sampler
            cursor.texture = mesh.texture
            aspect = float(window_size[0]) / float(window_size[1])
            camera_pos = [2, 2, 2]
            cursor.proj = glm.perspective(glm.radians(60), aspect, 0.1, 10)
            cursor.view = glm.lookAt(camera_pos, [0, 0, 0], [0, 1, 0])
            t = time.perf_counter() - self.timer
            offset = 0.2 * np.sin(2 * t)
            cursor.model = glm.translate([0, offset, 0]) * glm.rotate(t, [0, 1, 0])
            cursor.cameraPos = camera_pos

            pass_encoder.set_render_state(
                {
                    "viewports": [spy.Viewport.from_size(*window_size)],
                    "scissor_rects": [spy.ScissorRect.from_size(*window_size)],
                    "vertex_buffers": [
                        mesh.position_buffer,
                        mesh.normal_buffer,
                        mesh.uv_buffer,
                    ],
                    "index_buffer": mesh.index_buffer,
                    "index_format": mesh.index_format,
                }
            )

            pass_encoder.draw_indexed({"vertex_count": mesh.vertex_count})
