import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event

class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self):
        # ウィンドウの初期位置設定,RGBウィンドウを(x, y) = 左上(100, 100)に配置
        cv2.namedWindow('RGB')
        cv2.moveWindow('RGB', 100, 100)      
        # DepthウィンドウをRGBの右側(800, 100)に配置
        cv2.namedWindow('Depth')
        cv2.moveWindow('Depth', 600, 100)     

        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            confidence = self.session.get_confidence_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()  # Quaternion + world position

            print(intrinsic_mat)

            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

            # Postprocess it
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 位置姿勢（Pose）可視化
            pos_str = f"Pos: x={camera_pose.tx:.3f}, y={camera_pose.ty:.3f}, z={camera_pose.tz:.3f}"
            
            # 改行して回転を可視化
            rot_str1 = f"Rot: qx={camera_pose.qx:.3f}, qy={camera_pose.qy:.3f}"
            rot_str2 = f"     qz={camera_pose.qz:.3f}, qw={camera_pose.qw:.3f}"
            
            cv2.putText(rgb, pos_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(rgb, rot_str1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb, rot_str2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Show the RGBD Stream
            # 既にnamedWindowで作成済みなので、ここで画像が更新される
            cv2.imshow('RGB', rgb)
            cv2.imshow('Depth', depth)

            if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                cv2.imshow('Confidence', confidence * 100)

            cv2.waitKey(1)

            self.event.clear()


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()