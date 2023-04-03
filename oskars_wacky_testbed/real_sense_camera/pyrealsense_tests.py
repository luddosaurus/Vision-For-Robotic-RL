
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import ctypes
import math
import time

# Import OpenCV for easy image rendering
import cv2
# Import Numpy for easy array manipulation
import numpy as np
import pyglet
import pyglet.gl as gl
# First import the library
import pyrealsense2 as rs


def align_depth_to_color():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()



class AppState_1:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

def point_cloud_opencv():
    state = AppState_1()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()


    def mouse_cb(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)


    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


    def project(v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def view(v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


    def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = project(pt1.reshape(-1, 3))[0]
        p1 = project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
                view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
                view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                line3d(out, orig, view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            line3d(out, view(top_left), view(top_right), color)
            line3d(out, view(top_right), view(bottom_right), color)
            line3d(out, view(bottom_right), view(bottom_left), color)
            line3d(out, view(bottom_left), view(top_left), color)


    def pointcloud(out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]


    out = np.empty((h, w, 3), dtype=np.uint8)

    while True:
        # Grab camera data
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_frame = decimate.process(depth_frame)

            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # Render
        now = time.time()

        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

        cv2.imshow(state.WIN_NAME, out)
        key = cv2.waitKey(1)

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            state.decimate = (state.decimate + 1) % 3
            decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', out)

        if key == ord("e"):
            points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

    # Stop streaming
    pipeline.stop()

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

class AppState:

    def __init__(self, *args, **kwargs):
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, 1], np.float32)
        self.distance = 2
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 0
        self.scale = True
        self.attenuation = False
        self.color = True
        self.lighting = False
        self.postprocessing = False

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, 1

    @property
    def rotation(self):
        Rx = rotation_matrix((1, 0, 0), math.radians(-self.pitch))
        Ry = rotation_matrix((0, 1, 0), math.radians(-self.yaw))
        return np.dot(Ry, Rx).astype(np.float32)
    
def point_cloud_opengl():
    state = AppState()

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    other_stream, other_format = rs.stream.color, rs.format.rgb8
    config.enable_stream(other_stream, other_format, 30)

    # Start streaming
    pipeline.start(config)
    profile = pipeline.get_active_profile()

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()
    filters = [rs.disparity_transform(),
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.disparity_transform(False)]


    # pyglet
    window = pyglet.window.Window(
        config=gl.Config(
            double_buffer=True,
            samples=8  # MSAA
        ),
        resizable=True, vsync=True)
    keys = pyglet.window.key.KeyStateHandler()
    window.push_handlers(keys)


    def convert_fmt(fmt):
        """rs.format to pyglet format string"""
        return {
            rs.format.rgb8: 'RGB',
            rs.format.bgr8: 'BGR',
            rs.format.rgba8: 'RGBA',
            rs.format.bgra8: 'BGRA',
            rs.format.y8: 'L',
        }[fmt]


    # Create a VertexList to hold pointcloud data
    # Will pre-allocates memory according to the attributes below
    vertex_list = pyglet.graphics.vertex_list(
        w * h, 'v3f/stream', 't2f/stream', 'n3f/stream')
    # Create and allocate memory for our color data
    other_profile = rs.video_stream_profile(profile.get_stream(other_stream))

    image_w, image_h = w, h
    color_intrinsics = other_profile.get_intrinsics()
    color_w, color_h = color_intrinsics.width, color_intrinsics.height

    if state.color:
        image_w, image_h = color_w, color_h

    image_data = pyglet.image.ImageData(image_w, image_h, convert_fmt(
    other_profile.format()), (gl.GLubyte * (image_w * image_h * 3))())

    if (pyglet.version <  '1.4' ):
        # pyglet.clock.ClockDisplay has be removed in 1.4
        fps_display = pyglet.clock.ClockDisplay()
    else:
        fps_display = pyglet.window.FPSDisplay(window)


    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        w, h = map(float, window.get_size())

        if buttons & pyglet.window.mouse.LEFT:
            state.yaw -= dx * 0.5
            state.pitch -= dy * 0.5

        if buttons & pyglet.window.mouse.RIGHT:
            dp = np.array((dx / w, -dy / h, 0), np.float32)
            state.translation += np.dot(state.rotation, dp)

        if buttons & pyglet.window.mouse.MIDDLE:
            dz = dy * 0.01
            state.translation -= (0, 0, dz)
            state.distance -= dz


    def handle_mouse_btns(x, y, button, modifiers):
        state.mouse_btns[0] ^= (button & pyglet.window.mouse.LEFT)
        state.mouse_btns[1] ^= (button & pyglet.window.mouse.RIGHT)
        state.mouse_btns[2] ^= (button & pyglet.window.mouse.MIDDLE)


    window.on_mouse_press = window.on_mouse_release = handle_mouse_btns


    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        dz = scroll_y * 0.1
        state.translation -= (0, 0, dz)
        state.distance -= dz


    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.R:
            state.reset()

        if symbol == pyglet.window.key.P:
            state.paused ^= True

        if symbol == pyglet.window.key.D:
            state.decimate = (state.decimate + 1) % 3
            decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

        if symbol == pyglet.window.key.C:
            state.color ^= True

        if symbol == pyglet.window.key.Z:
            state.scale ^= True

        if symbol == pyglet.window.key.X:
            state.attenuation ^= True

        if symbol == pyglet.window.key.L:
            state.lighting ^= True

        if symbol == pyglet.window.key.F:
            state.postprocessing ^= True

        if symbol == pyglet.window.key.S:
            pyglet.image.get_buffer_manager().get_color_buffer().save('out.png')

        if symbol == pyglet.window.key.Q:
            window.close()


    window.push_handlers(on_key_press)


    def axes(size=1, width=1):
        """draw 3d axes"""
        gl.glLineWidth(width)
        pyglet.graphics.draw(6, gl.GL_LINES,
                            ('v3f', (0, 0, 0, size, 0, 0,
                                    0, 0, 0, 0, size, 0,
                                    0, 0, 0, 0, 0, size)),
                            ('c3f', (1, 0, 0, 1, 0, 0,
                                    0, 1, 0, 0, 1, 0,
                                    0, 0, 1, 0, 0, 1,
                                    ))
                            )


    def frustum(intrinsics):
        """draw camera's frustum"""
        w, h = intrinsics.width, intrinsics.height
        batch = pyglet.graphics.Batch()

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                batch.add(2, gl.GL_LINES, None, ('v3f', [0, 0, 0] + p))
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            batch.add(2, gl.GL_LINES, None, ('v3f', top_left + top_right))
            batch.add(2, gl.GL_LINES, None, ('v3f', top_right + bottom_right))
            batch.add(2, gl.GL_LINES, None, ('v3f', bottom_right + bottom_left))
            batch.add(2, gl.GL_LINES, None, ('v3f', bottom_left + top_left))

        batch.draw()


    def grid(size=1, n=10, width=1):
        """draw a grid on xz plane"""
        gl.glLineWidth(width)
        s = size / float(n)
        s2 = 0.5 * size
        batch = pyglet.graphics.Batch()

        for i in range(0, n + 1):
            x = -s2 + i * s
            batch.add(2, gl.GL_LINES, None, ('v3f', (x, 0, -s2, x, 0, s2)))
        for i in range(0, n + 1):
            z = -s2 + i * s
            batch.add(2, gl.GL_LINES, None, ('v3f', (-s2, 0, z, s2, 0, z)))

        batch.draw()


    @window.event
    def on_draw():
        window.clear()

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)

        width, height = window.get_size()
        gl.glViewport(0, 0, width, height)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(60, width / float(height), 0.01, 20)

        gl.glMatrixMode(gl.GL_TEXTURE)
        gl.glLoadIdentity()
        # texcoords are [0..1] and relative to top-left pixel corner, add 0.5 to center
        gl.glTranslatef(0.5 / image_data.width, 0.5 / image_data.height, 0)
        image_texture = image_data.get_texture()
        # texture size may be increased by pyglet to a power of 2
        tw, th = image_texture.owner.width, image_texture.owner.height
        gl.glScalef(image_data.width / float(tw),
                    image_data.height / float(th), 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gl.gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

        gl.glTranslatef(0, 0, state.distance)
        gl.glRotated(state.pitch, 1, 0, 0)
        gl.glRotated(state.yaw, 0, 1, 0)

        if any(state.mouse_btns):
            axes(0.1, 4)

        gl.glTranslatef(0, 0, -state.distance)
        gl.glTranslatef(*state.translation)

        gl.glColor3f(0.5, 0.5, 0.5)
        gl.glPushMatrix()
        gl.glTranslatef(0, 0.5, 0.5)
        grid()
        gl.glPopMatrix()

        psz = max(window.get_size()) / float(max(w, h)) if state.scale else 1
        gl.glPointSize(psz)
        distance = (0, 0, 1) if state.attenuation else (1, 0, 0)
        gl.glPointParameterfv(gl.GL_POINT_DISTANCE_ATTENUATION,
                            (gl.GLfloat * 3)(*distance))

        if state.lighting:
            ldir = [0.5, 0.5, 0.5]  # world-space lighting
            ldir = np.dot(state.rotation, (0, 0, 1))  # MeshLab style lighting
            ldir = list(ldir) + [0]  # w=0, directional light
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                        (gl.GLfloat * 3)(1.0, 1.0, 1.0))
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                        (gl.GLfloat * 3)(0.75, 0.75, 0.75))
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_NORMALIZE)
            gl.glEnable(gl.GL_LIGHTING)

        gl.glColor3f(1, 1, 1)
        texture = image_data.get_texture()
        gl.glEnable(texture.target)
        gl.glBindTexture(texture.target, texture.id)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # comment this to get round points with MSAA on
        gl.glEnable(gl.GL_POINT_SPRITE)

        if not state.scale and not state.attenuation:
            gl.glDisable(gl.GL_MULTISAMPLE)  # for true 1px points with MSAA on
        vertex_list.draw(gl.GL_POINTS)
        gl.glDisable(texture.target)
        if not state.scale and not state.attenuation:
            gl.glEnable(gl.GL_MULTISAMPLE)

        gl.glDisable(gl.GL_LIGHTING)

        gl.glColor3f(0.25, 0.25, 0.25)
        frustum(depth_intrinsics)
        axes()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_TEXTURE)
        gl.glLoadIdentity()
        gl.glDisable(gl.GL_DEPTH_TEST)

        fps_display.draw()


    def run(dt):
        global w, h
        window.set_caption("RealSense (%dx%d) %dFPS (%.2fms) %s" %
                        (w, h, 0 if dt == 0 else 1.0 / dt, dt * 1000,
                            "PAUSED" if state.paused else ""))

        if state.paused:
            return

        success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
        if not success:
            return

        depth_frame = frames.get_depth_frame().as_video_frame()
        other_frame = frames.first(other_stream).as_video_frame()

        depth_frame = decimate.process(depth_frame)

        if state.postprocessing:
            for f in filters:
                depth_frame = f.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        color_image = np.asanyarray(other_frame.get_data())

        colorized_depth = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(colorized_depth.get_data())

        if state.color:
            mapped_frame, color_source = other_frame, color_image
        else:
            mapped_frame, color_source = colorized_depth, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # handle color source or size change
        fmt = convert_fmt(mapped_frame.profile.format())
        global image_data

        if (image_data.format, image_data.pitch) != (fmt, color_source.strides[0]):
            if state.color:
                global color_w, color_h
                image_w, image_h = color_w, color_h
            else:
                image_w, image_h = w, h

            empty = (gl.GLubyte * (image_w * image_h * 3))()
            image_data = pyglet.image.ImageData(image_w, image_h, fmt, empty)

        # copy image data to pyglet
        image_data.set_data(fmt, color_source.strides[0], color_source.ctypes.data)

        verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
        texcoords = np.asarray(points.get_texture_coordinates(2))

        if len(vertex_list.vertices) != verts.size:
            vertex_list.resize(verts.size // 3)
            # need to reassign after resizing
            vertex_list.vertices = verts.ravel()
            vertex_list.tex_coords = texcoords.ravel()

        # copy our data to pre-allocated buffers, this is faster than assigning...
        # pyglet will take care of uploading to GPU
        def copy(dst, src):
            """copy numpy array to pyglet array"""
            # timeit was mostly inconclusive, favoring slice assignment for safety
            np.array(dst, copy=False)[:] = src.ravel()
            # ctypes.memmove(dst, src.ctypes.data, src.nbytes)

        copy(vertex_list.vertices, verts)
        copy(vertex_list.tex_coords, texcoords)

        if state.lighting:
            # compute normals
            dy, dx = np.gradient(verts, axis=(0, 1))
            n = np.cross(dx, dy)

            # can use this, np.linalg.norm or similar to normalize, but OpenGL can do this for us, see GL_NORMALIZE above
            # norm = np.sqrt((n*n).sum(axis=2, keepdims=True))
            # np.divide(n, norm, out=n, where=norm != 0)

            # import cv2
            # n = cv2.bilateralFilter(n, 5, 1, 1)

            copy(vertex_list.normals, n)

        if keys[pyglet.window.key.E]:
            points.export_to_ply('./out.ply', mapped_frame)


    pyglet.clock.schedule(run)

    try:
        pyglet.app.run()
    finally:
        pipeline.stop()



def main():
    align_depth_to_color()
    #point_cloud_opencv()

    #------------------------------does not work!------------------------
    # point_cloud_opengl()
    #--------------------------------------------------------------------

if __name__ == '__main__':
    main()