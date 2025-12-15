import json
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import liblzfse
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data import Dataset

from dataloaders.scannet_200_classes import CLASS_LABELS_200


class R3DSemanticDataset_lessram(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
        sample_freq: int = 5,  # デフォルト値を30に設定（メモリ不足対策）
    ):
        # メモリ節約のため、ここで間引き間隔を保持
        self.sample_freq = sample_freq
        print(f"Sampling frequency set to: {self.sample_freq}")

        if path.endswith((".zip", ".r3d")):
            self._path = ZipFile(path)
        else:
            self._path = Path(path)

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200

        self._reshaped_depth = []
        self._reshaped_conf = []
        self._depth_images = []
        self._rgb_images = []
        self._confidences = []

        self._metadata = self._read_metadata()
        self.global_xyzs = []
        self.global_pcds = []
        
        # データのロード（間引き処理込み）
        self._load_data()
        
        # 間引かれたデータに対してのみ実行されるため高速かつ軽量
        self._reshape_all_depth_and_conf()
        self.calculate_all_global_xyzs()

    def _read_metadata(self):
        with self._path.open("metadata", "r") as f:
            metadata_dict = json.load(f)

        # Now figure out the details from the metadata dict.
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.fps = metadata_dict["fps"]
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.image_size = (self.rgb_width, self.rgb_height)
        
        # 全ポーズデータを取得
        all_poses = np.array(metadata_dict["poses"])
        
        # 【重要】ここでデータを間引くためのインデックスを作成
        # range(開始, 終了, ステップ) を使うことで読み込むフレームを決める
        total_raw_images = len(all_poses)
        self.indices = list(range(0, total_raw_images, self.sample_freq))
        
        # ポーズ情報も間引いたものだけにする（これでメモリと整合性が取れる）
        self.poses = all_poses[self.indices]
        
        self.init_pose = np.array(metadata_dict["initPose"])
        
        # total_imagesは「学習に使う実際の枚数」にする
        self.total_images = len(self.poses)
        
        print(f"Total frames: {total_raw_images} -> Sampling {self.total_images} frames.")

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        with self._path.open(filepath, "r") as image_file:
            return np.asarray(Image.open(image_file))

    def load_depth(self, filepath):
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img: np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.float32)

        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        elif depth_img.shape[0] == 640 * 480:
            depth_img = depth_img.reshape((640, 480))  # For High Quality Depth
        else:
            depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        return depth_img

    def load_conf(self, filepath):
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
        
        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        elif depth_img.shape[0] == 640 * 480:
            depth_img = depth_img.reshape((640, 480))  # For High Quality Depth
        else:
            depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        return depth_img

    def _load_data(self):
        assert self.fps  # Make sure metadata is read correctly first.
        
        # 【重要】self.indices（間引き後のインデックス）を使ってループする
        # これにより、必要なファイルだけを物理的に読み込む
        for i in tqdm.tqdm(self.indices, desc="Loading data (Subsampled)"):
            # i は元のファイル番号（例: 0, 30, 60...）
            rgb_filepath = f"rgbd/{i}.jpg"
            depth_filepath = f"rgbd/{i}.depth"
            conf_filepath = f"rgbd/{i}.conf"

            depth_img = self.load_depth(depth_filepath)
            
            # Try loading confidence, otherwise fill with 2 (high confidence)
            try:
                confidence = self.load_conf(conf_filepath)
            except KeyError:
                confidence = np.full(depth_img.shape, 2, dtype=np.uint8)
                
            rgb_img = self.load_image(rgb_filepath)

            # リストに追加（メモリに乗るのはこれだけ）
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)
            self._confidences.append(confidence)

    def _reshape_all_depth_and_conf(self):
        # self.posesは既に間引かれているので、ループ回数は少なくなっている
        for index in tqdm.trange(len(self.poses), desc="Upscaling depth and conf"):
            depth_image = self._depth_images[index]
            # Upscale depth image.
            pil_img = Image.fromarray(depth_image)
            reshaped_img = pil_img.resize((self.rgb_width, self.rgb_height), Image.Resampling.NEAREST)
            reshaped_img = np.asarray(reshaped_img)
            self._reshaped_depth.append(reshaped_img)

            # Upscale confidence as well
            confidence = self._confidences[index]
            conf_img = Image.fromarray(confidence)
            reshaped_conf = conf_img.resize((self.rgb_width, self.rgb_height), Image.Resampling.NEAREST)
            reshaped_conf = np.asarray(reshaped_conf)
            self._reshaped_conf.append(reshaped_conf)

    def get_global_xyz(self, index, depth_scale=1000.0, only_confident=True):
        reshaped_img = np.copy(self._reshaped_depth[index])

        # If only confident, replace not confident points with nans
        if only_confident:
            reshaped_img[self._reshaped_conf[index] != 2] = np.nan
            # valid_mask = (self._reshaped_conf[index] == 2) & (reshaped_img < 3.0)
            # reshaped_img[~valid_mask] = np.nan

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        # Flip the pcd
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        extrinsic_matrix = np.eye(4)
        
        # self.posesはスライス済みなので、indexがそのまま対応する
        qx, qy, qz, qw, px, py, pz = self.poses[index]
        extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]
        pcd.transform(extrinsic_matrix)

        # Now transform everything by init pose.
        init_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.init_pose
        init_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        init_matrix[:3, -1] = [px, py, pz]
        pcd.transform(init_matrix)

        return pcd

    def calculate_all_global_xyzs(self, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            self.global_xyzs.append(global_xyz)
            self.global_pcds.append(global_xyz_pcd)
        return self.global_xyzs, self.global_pcds

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],
            "conf": self._reshaped_conf[idx],
        }
        return result