import slangpy as spy
from Mesh import Mesh
from MeshRenderer import MeshRenderer


class App:
    def __init__(self):
        self.window = spy.Window(1600, 1200, "obj-viewer", resizable=True)

        self.device = spy.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": ["assets/shaders"]},
        )

        self.mesh = Mesh("assets/models/monkey.obj", self.device)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(self.window.width, self.window.height)
        self.mesh_renderer = MeshRenderer(self.device, self.surface.config.format)
        self.window.on_mouse_event = self.handle_mouse_event
        self.window.on_resize = self.handle_resize
        self.create_depth_texture()
        self.setup_ui()
        self.dirty = False
        self.surface_texture = None

    def setup_ui(self):
        self.ui = spy.ui.Context(self.device)

        window = spy.ui.Window(
            self.ui.screen, "Settings", spy.float2(10, 10), spy.float2(300, 100)
        )
        
        spy.ui.Text(window, "Hello, World!")

    def handle_resize(self, width, height):
        self.dirty = True

    def create_depth_texture(self):
        self.depth_texture = self.device.create_texture(
            format=spy.Format.d32_float,
            width=self.window.width,
            height=self.window.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    def resize(self):
        del self.depth_texture
        del self.surface_texture
        self.device.wait()
        self.surface.configure(self.window.width, self.window.height)
        self.create_depth_texture()

    def handle_mouse_event(self, event):
        self.ui.handle_mouse_event(event)

    def run(self):
        while not self.window.should_close():
            self.window.process_events()
            self.ui.process_events()

            if self.dirty:
                self.resize()
                self.dirty = False

            self.surface_texture = self.surface.acquire_next_image()

            if not self.surface_texture:
                continue

            command_encoder = self.device.create_command_encoder()
            window_size = (self.window.width, self.window.height)

            self.mesh_renderer.render(
                command_encoder,
                self.mesh,
                window_size,
                self.surface_texture,
                self.depth_texture,
            )

            self.ui.new_frame(*window_size)
            self.ui.render(self.surface_texture, command_encoder)
            self.device.submit_command_buffer(command_encoder.finish())
            self.surface.present()


if __name__ == "__main__":
    app = App()
    app.run()
