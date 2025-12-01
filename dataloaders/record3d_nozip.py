import json
from pathlib import Path
from typing import List, Optional
# ZipFileは使用しないため削除
# from zipfile import ZipFile

import liblzfse
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data import Dataset

from dataloaders.scannet_200_classes import CLASS_LABELS_200


class R3DSemanticDataset_nozip(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
    ):
        # パスをディレクトリとして扱うように変更
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

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
        
        # データを読み込む
        self._load_data()
        
        # 読み込み後のデータ処理
        self._reshape_all_depth_and_conf()
        self.calculate_all_global_xyzs()

    def _read_metadata(self):
        # ディレクトリ内のmetadataファイルを読み込む
        metadata_path = self._path / "metadata"
        with metadata_path.open("r") as f:
            metadata_dict = json.load(f)

        # メタデータ辞書から詳細を取得する
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.fps = metadata_dict["fps"]
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.image_size = (self.rgb_width, self.rgb_height)
        self.poses = np.array(metadata_dict["poses"])
        self.init_pose = np.array(metadata_dict["initPose"])
        self.total_images = len(self.poses)

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}
        return metadata_dict

    def load_image(self, filepath):
        # ディレクトリ内の画像ファイルを読み込む
        full_path = self._path / filepath
        return np.asarray(Image.open(full_path))

    def load_depth(self, filepath):
        # ディレクトリ内の深度ファイルをバイナリモード("rb")で読み込む
        full_path = self._path / filepath
        with full_path.open("rb") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img: np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.float32)

        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))
        elif depth_img.shape[0] == 640 * 480:
            depth_img = depth_img.reshape((640, 480))
        else:
            depth_img = depth_img.reshape((256, 192))
        return depth_img

    def load_conf(self, conf_filepath):
        # ファイルが存在するか確認は呼び出し元で行う前提、あるいはここでチェック
        full_path = self._path / conf_filepath
        
        if not full_path.exists():
             return None

        with open(full_path, 'rb') as f:
            raw_bytes = f.read()

        if len(raw_bytes) == 0:
            # print(f"[Warning] Empty file found: {conf_filepath}")
            return None

        try:
            decompressed_bytes = liblzfse.decompress(raw_bytes)
        except Exception as e:
            print(f"!! LZFSE Error on file: {conf_filepath}")
            print(f"!! Error details: {e}")
            return None

        depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
        
        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))
        elif depth_img.shape[0] == 640 * 480:
            depth_img = depth_img.reshape((640, 480))
        else:
            depth_img = depth_img.reshape((256, 192))
        return depth_img

    def _load_data(self):
        assert self.fps  # 最初にメタデータが正しく読み込まれているか確認
        
        valid_indices = [] # 正常に読み込めたフレームのインデックスを保存
        
        for i in tqdm.trange(self.total_images, desc="Loading data"):
            rgb_rel_path = f"rgbd/{i}.jpg"
            depth_rel_path = f"rgbd/{i}.depth"
            conf_rel_path = f"rgbd/{i}.conf"

            # 【重要修正】ファイルが存在しない場合はスキップする
            if not (self._path / rgb_rel_path).exists() or not (self._path / depth_rel_path).exists():
                # print(f"Skipping frame {i}: Files missing.") # ログが多すぎる場合はコメントアウト
                continue

            try:
                # Depth読み込み
                depth_img = self.load_depth(depth_rel_path)
                
                # RGB読み込み
                rgb_img = self.load_image(rgb_rel_path)

                # Confidence読み込み（失敗したらデフォルト値）
                confidence = self.load_conf(conf_rel_path)
                if confidence is None:
                    confidence = np.full(depth_img.shape, 2, dtype=np.uint8)

            except Exception as e:
                print(f"Error loading frame {i}: {e}. Skipping.")
                continue

            # データリストに追加
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)
            self._confidences.append(confidence)
            
            # 成功したインデックスを記録
            valid_indices.append(i)

        # 【重要修正】メタデータのposes配列を、実際に読み込めたフレームだけにフィルタリングする
        # これをしないと、__getitem__でインデックスがずれてエラーになります
        if len(valid_indices) < self.total_images:
            print(f"[Info] Loaded {len(valid_indices)} frames out of {self.total_images}. Truncating metadata.")
            self.poses = self.poses[valid_indices]
            self.total_images = len(self.poses)

    def _reshape_all_depth_and_conf(self):
        # self.posesの長さ（実際に読み込まれた枚数）に合わせてループ
        for index in tqdm.trange(len(self.poses), desc="Upscaling depth and conf"):
            depth_image = self._depth_images[index]
            # 深度画像のアップスケール
            pil_img = Image.fromarray(depth_image)
            reshaped_img = pil_img.resize((self.rgb_width, self.rgb_height), Image.Resampling.NEAREST)
            reshaped_img = np.asarray(reshaped_img)
            self._reshaped_depth.append(reshaped_img)

            # 信頼度のアップスケール
            confidence = self._confidences[index]
            conf_img = Image.fromarray(confidence)
            reshaped_conf = conf_img.resize((self.rgb_width, self.rgb_height), Image.Resampling.NEAREST)
            reshaped_conf = np.asarray(reshaped_conf)
            self._reshaped_conf.append(reshaped_conf)

    def get_global_xyz(self, index, depth_scale=1000.0, only_confident=True):
        reshaped_img = np.copy(self._reshaped_depth[index])

        # 高信頼度の点のみを使用する場合、それ以外をNaNにする
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
        # 点群を反転
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        extrinsic_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.poses[index]
        extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]
        pcd.transform(extrinsic_matrix)

        # 初期ポーズで全体を変換
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