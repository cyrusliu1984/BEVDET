import os
import cv2
import torch
import math
import time
import json
import argparse
import numpy as np
import zmq  # 【新增】引入ZeroMQ
import pickle  # 【新增】引入pickle
from pyquaternion import Quaternion
from mmcv import Config
from mmcv.parallel import DataContainer
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox import Box3DMode
from mmdet3d.models import build_model
from mmcv.parallel import MMDataParallel
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ===================== 0. 全局配置 & 核心常量（仅新增点云缩放因子配置） =====================
CONFIG_PATH = "configs/bevdet/bevdet-r50.py"
WEIGHT_PATH = "/workspace/BEV/ckpt/bevdet-dev2.1/bevdet-r50.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.1
VIS_THRED = 0.3  # 对齐demo默认值
CAM_NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# 【新增】ZeroMQ配置
VIEWS = CAM_NAMES 
ZMQ_SUB_ADDR = "tcp://127.0.0.1:5555"  # ZeroMQ服务端地址
# ========== 原有配置保持不变 ==========
NUSCENES_IMG_SIZE = (1600, 900)  # (宽, 高)
VIS_SAVE_PATH = "./bevdet_vis_results"  # 最终图像保存目录
SHOW_RANGE = 50  
CANVA_SIZE = 1000  
SCALE_FACTOR = 4  
# 【新增】点云显示范围缩放因子（核心新增配置）
POINT_RANGE_SCALE = 2.0  # >1扩大显示范围，<1缩小，默认1.0不变

# Pipeline核心配置
DATA_CONFIG = {
    'cams': CAM_NAMES,
    'Ncams': 6,
    'input_size': (256, 704),  # (h, w)
    'src_size': (900, 1600),    # (h, w)
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# sensor2ego 4x4变换矩阵（6个相机）
SENSOR2EGO_MATRICES = torch.tensor([
    [[ 8.2076e-01, -3.4144e-04,  5.7128e-01,  1.5239e+00],
     [-5.7127e-01,  3.2195e-03,  8.2075e-01,  4.9463e-01],
     [-2.1195e-03, -9.9999e-01,  2.4474e-03,  1.5093e+00],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # CAM_FRONT_LEFT
    [[ 5.6848e-03, -5.6367e-03,  9.9997e-01,  1.7008e+00],
     [-9.9998e-01, -8.3712e-04,  5.6801e-03,  1.5946e-02],
     [ 8.0507e-04, -9.9998e-01, -5.6413e-03,  1.5110e+00],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # CAM_FRONT
    [[-8.3293e-01, -9.9460e-06,  5.5338e-01,  1.5508e+00],
     [-5.5330e-01,  1.6379e-02, -8.3282e-01, -4.9340e-01],
     [-9.0554e-03, -9.9987e-01, -1.3648e-02,  1.4957e+00],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # CAM_FRONT_RIGHT
    [[ 9.4776e-01,  8.6657e-03, -3.1887e-01,  1.0357e+00],
     [ 3.1896e-01, -1.3976e-02,  9.4766e-01,  4.8480e-01],
     [ 3.7556e-03, -9.9986e-01, -1.6010e-02,  1.5910e+00],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # CAM_BACK_LEFT
    [[ 2.4217e-03, -1.6754e-02, -9.9986e-01,  2.8326e-02],
     [ 9.9999e-01, -3.9591e-03,  2.4884e-03,  3.4514e-03],
     [-4.0002e-03, -9.9985e-01,  1.6744e-02,  1.5791e+00],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],  # CAM_BACK
    [[-9.3478e-01,  1.5876e-02, -3.5488e-01,  1.0149e+00],
     [ 3.5507e-01,  1.1370e-02, -9.3477e-01, -4.8057e-01],
     [-1.0805e-02, -9.9981e-01, -1.6266e-02,  1.5624e+00],
     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]   # CAM_BACK_RIGHT
], dtype=torch.float32)

# 核心常量（保持不变）
COLOR_MAP = {0: (255, 255, 0), 1: (0, 255, 255)}
DRAW_BOXES_INDEXES_BEV = [(0, 1), (1, 2), (2, 3), (3, 0)]
DRAW_BOXES_INDEXES_IMG_VIEW = [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

# ===================== 【新增】点云旋转函数（完全复用bag处理代码） =====================
def rotate_point_cloud_z(points_np, direction="clockwise"):
    """
    对点云绕 Z 轴旋转 90 度（与bag提取代码逻辑完全一致）
    :param points_np: 原始点云数组，shape=(N,>=3)，至少包含 [x, y, z]
    :param direction: 旋转方向，可选 "clockwise"（顺时针）或 "counterclockwise"（逆时针）
    :return: 旋转后的点云数组
    """
    # 提取 x, y, z 坐标（保留所有额外维度如intensity）
    x = points_np[:, 0]
    y = points_np[:, 1]
    z = points_np[:, 2]
    # 保留x/y/z之外的维度（如intensity）
    extra_dims = points_np[:, 3:] if points_np.shape[1] > 3 else None

    # 绕 Z 轴旋转 90 度（右手坐标系：X向右，Y向前，Z向上）
    if direction == "clockwise":
        # 顺时针旋转 90 度：x' = y, y' = -x（与bag代码完全一致）
        new_x = -x
        new_y = -y
    elif direction == "counterclockwise":
        # 逆时针旋转 90 度：x' = -y, y' = x
        new_x = -x
        new_y = -y
    else:
        raise ValueError("direction 只能是 'clockwise' 或 'counterclockwise'")

    # 重组点云（保留原始所有维度）
    rotated_points = np.column_stack([new_x, new_y, z])
    if extra_dims is not None:
        rotated_points = np.column_stack([rotated_points, extra_dims])
    
    return rotated_points.astype(np.float32)

# ===================== 1. 工具函数（完全保持不变） =====================
def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid

def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int32)
    diff = (gray - rank / num_rank) * num_rank
    return tuple((colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())

def lidar2img(points_lidar, camera_info):
    points_lidar_homogeneous = np.concatenate([
        points_lidar, 
        np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)
    ], axis=1)
    
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camera_info['sensor2lidar_translation']
    
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = lidar2camera @ points_lidar_homogeneous.T
    points_camera = points_camera_homogeneous.T[:, :3]
    
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, 2] > 0.5, valid)
    
    non_zero = points_camera[:, 2] != 0
    points_camera[non_zero] = points_camera[non_zero] / points_camera[non_zero, 2:3]
    points_camera[~non_zero] = 0
    
    camera2img = camera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    
    return points_img, valid

def get_sensor_transforms(cam_name):
    cam_index = CAM_NAMES.index(cam_name)
    sensor2ego = SENSOR2EGO_MATRICES[cam_index].cpu().numpy()
    
    sensor2lidar_rotation = sensor2ego[:3, :3]
    sensor2lidar_translation = sensor2ego[:3, 3]
    
    cam_intrinsic = get_cam_intrinsic(cam_name).cpu().numpy()
    
    return {
        'sensor2lidar_rotation': sensor2lidar_rotation,
        'sensor2lidar_translation': sensor2lidar_translation,
        'cam_intrinsic': cam_intrinsic,
        'sensor2ego': sensor2ego
    }

def get_cam_intrinsic(cam_name):
    intrinsics = {
        'CAM_FRONT_LEFT': np.array([[1272.59795, 0, 826.615493],
                                    [0, 1272.59795, 479.751654],
                                    [0, 0, 1]]),
        'CAM_FRONT': np.array([[1266.41720, 0, 816.267020],
                               [0, 1266.41720, 491.507066],
                               [0, 0, 1]]),
        'CAM_FRONT_RIGHT': np.array([[1260.84744, 0, 807.968245],
                                     [0, 1260.84744, 495.334427],
                                     [0, 0, 1]]),
        'CAM_BACK_LEFT': np.array([[1256.74148, 0, 792.112574],
                                   [0, 1256.74148, 492.775747],
                                   [0, 0, 1]]),
        'CAM_BACK': np.array([[809.22099057, 0, 829.21960033],
                              [0, 809.22099057, 481.77842385],
                              [0, 0, 1]]),
        'CAM_BACK_RIGHT': np.array([[1259.51374, 0, 807.252905],
                                    [0, 1259.51374, 501.195799],
                                    [0, 0, 1]])
    }
    return torch.Tensor(intrinsics[cam_name])

# ===================== 2. 【修改】ZeroMQ数据接收模块（新增点云旋转） =====================
class ZMQDataReceiver:
    """ZeroMQ数据接收类（替换原LocalDataLoader）"""
    def __init__(self, zmq_addr, rotate_z_direction="clockwise"):
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # 订阅所有消息
        self.socket.connect(zmq_addr)
        self.socket.RCVTIMEO = 5000  # 5秒超时
        # 【新增】点云旋转配置
        self.rotate_z_direction = rotate_z_direction
        print(f"✅ ZeroMQ客户端已连接到：{zmq_addr}")
        print(f"✅ 点云绕Z轴旋转配置：{self.rotate_z_direction} 90度")

    def receive_data(self):
        """接收并解析ZeroMQ数据（新增点云旋转逻辑）"""
        try:
            # 接收序列化数据
            serialized_data = self.socket.recv()
            data = pickle.loads(serialized_data)
            
            # 验证必要字段
            required_fields = ["timestamp", "image", "lidar"]
            if not all(k in data for k in required_fields):
                raise ValueError(f"数据缺少必要字段，当前字段：{list(data.keys())}")
            
            # 验证数据格式
            img = data["image"]
            lidar = data["lidar"]
            
            # 图像尺寸验证/调整
            if img.shape[:2] != (NUSCENES_IMG_SIZE[1], NUSCENES_IMG_SIZE[0]):
                print(f"⚠️  图像尺寸不匹配，自动调整：{img.shape} → {(NUSCENES_IMG_SIZE[1], NUSCENES_IMG_SIZE[0])}")
                img = cv2.resize(img, (NUSCENES_IMG_SIZE[0], NUSCENES_IMG_SIZE[1]))
            
            # 点云格式验证（至少包含x,y,z）
            if len(lidar.shape) != 2 or lidar.shape[1] < 3:
                raise ValueError(f"点云格式错误，期望(N,>=3)，实际{lidar.shape}")
            
            # ===================== 【核心修改】点云绕Z轴旋转90度 =====================
            print(f"🔄 对点云进行绕Z轴{self.rotate_z_direction}旋转90度处理...")
            lidar_rotated = rotate_point_cloud_z(lidar, direction=self.rotate_z_direction)
            print(f"✅ 点云旋转完成：原始shape{lidar.shape} → 旋转后shape{lidar_rotated.shape}")
            
            print(f"✅ 接收数据成功 | 时间戳：{data['timestamp']} | 图像尺寸：{img.shape} | 点云点数：{lidar_rotated.shape[0]}")
            return {
                "timestamp": data["timestamp"],
                "image": img,          # 前视图像 (900,1600,3)
                "lidar": lidar_rotated[:, :3]  # 仅保留x,y,z（旋转后）
            }
        except zmq.Again:
            raise TimeoutError("❌ ZeroMQ接收超时（5秒）")
        except Exception as e:
            raise RuntimeError(f"❌ 接收/解析数据失败：{str(e)}")

    def close(self):
        """关闭连接"""
        self.socket.close()
        self.zmq_context.term()

# ===================== 3. 绘制函数（核心修改：添加点云缩放因子） =====================
def draw_3d_box_on_img_ref(img, bbox_3d, camera_info):
    if len(bbox_3d) == 0:
        return img
    
    corners_lidar = bbox_3d.corners.numpy().reshape(-1, 3)
    corners_img, valid = lidar2img(corners_lidar, camera_info)
    
    valid = np.logical_and(
        valid,
        check_point_in_img(corners_img, img.shape[0], img.shape[1]))
    valid = valid.reshape(-1, 8)
    corners_img = corners_img.reshape(-1, 8, 2).astype(np.int32)
    
    pred_flag = np.ones((valid.shape[0],), dtype=np.bool)
    for aid in range(valid.shape[0]):
        for index in DRAW_BOXES_INDEXES_IMG_VIEW:
            if valid[aid, index[0]] and valid[aid, index[1]]:
                cv2.line(
                    img,
                    tuple(corners_img[aid, index[0]]),
                    tuple(corners_img[aid, index[1]]),
                    color=COLOR_MAP[int(pred_flag[aid])],
                    thickness=SCALE_FACTOR,
                    lineType=cv2.LINE_AA
                )
    return img

def draw_bev_view(lidar_points, valid_bboxes, scores, point_scale=POINT_RANGE_SCALE):  # 新增缩放参数
    canvas = np.zeros((int(CANVA_SIZE), int(CANVA_SIZE), 3), dtype=np.uint8)
    
    # 绘制雷达点云（核心修改：应用缩放因子）
    if len(lidar_points) > 0:
        lidar_xyz = lidar_points.copy()
        lidar_xyz[:, 1] = -lidar_xyz[:, 1]
        # 应用点云显示范围缩放因子（仅修改这一行）
        lidar_xyz[:, :2] = (lidar_xyz[:, :2] * point_scale + SHOW_RANGE) / SHOW_RANGE / 2.0 * CANVA_SIZE
        
        for p in lidar_xyz:
            if check_point_in_img(p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                color = depth2color(p[2])
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])),
                    radius=0,
                    color=color,
                    thickness=1)
    
    # 绘制3D框BEV投影（完全不变，不受缩放影响）
    if len(valid_bboxes) > 0:
        corners_lidar = valid_bboxes.corners.numpy().reshape(-1, 8, 3)
        # corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        x = corners_lidar[:, :, 0].copy()
        y = corners_lidar[:, :, 1].copy()
        corners_lidar[:, :, 0] = -y
        corners_lidar[:, :, 1] = -x
        
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = (bottom_corners_bev + SHOW_RANGE) / SHOW_RANGE / 2.0 * CANVA_SIZE
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        
        center_canvas = (center_bev + SHOW_RANGE) / SHOW_RANGE / 2.0 * CANVA_SIZE
        center_canvas = center_canvas.astype(np.int32)
        head_canvas = (head_bev + SHOW_RANGE) / SHOW_RANGE / 2.0 * CANVA_SIZE
        head_canvas = head_canvas.astype(np.int32)
        
        scores_np = scores.cpu().numpy() if len(scores) > 0 else np.array([])
        sort_ids = np.argsort(scores_np) if len(scores_np) > 0 else []
        
        pred_flag = np.ones((len(valid_bboxes),), dtype=np.bool)
        for rid in sort_ids:
            score = scores_np[rid] if len(scores_np) > 0 else 0
            if score < VIS_THRED and pred_flag[rid]:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = COLOR_MAP[int(pred_flag[rid])]
            
            for index in DRAW_BOXES_INDEXES_BEV:
                cv2.line(
                    canvas,
                    tuple(bottom_corners_bev[rid, index[0]]),
                    tuple(bottom_corners_bev[rid, index[1]]),
                    [int(color[0] * score), int(color[1] * score), int(color[2] * score)],
                    thickness=1,
                    lineType=cv2.LINE_AA)
            
            cv2.line(
                canvas,
                tuple(center_canvas[rid]),
                tuple(head_canvas[rid]),
                [int(color[0] * score), int(color[1] * score), int(color[2] * score)],
                1,
                lineType=cv2.LINE_AA)
    
    return canvas

def fuse_img_bev(imgs, bev_view):
    img_width = NUSCENES_IMG_SIZE[0]
    img_height = NUSCENES_IMG_SIZE[1]
    fuse_width = int(img_width / SCALE_FACTOR * 3)
    fuse_height = int(img_height / SCALE_FACTOR * 2 + CANVA_SIZE)
    
    fuse_img = np.zeros((fuse_height, fuse_width, 3), dtype=np.uint8)
    
    top_imgs = np.concatenate(imgs[:3], axis=1)
    top_imgs_resized = cv2.resize(top_imgs, (fuse_width, int(img_height / SCALE_FACTOR)))
    fuse_img[:int(img_height / SCALE_FACTOR), :, :] = top_imgs_resized
    
    bottom_imgs = np.concatenate([img[:, ::-1, :] for img in imgs[3:]], axis=1)
    bottom_imgs_resized = cv2.resize(bottom_imgs, (fuse_width, int(img_height / SCALE_FACTOR)))
    fuse_img[int(img_height / SCALE_FACTOR) + CANVA_SIZE:, :, :] = bottom_imgs_resized
    
    bev_x_start = int((fuse_width - CANVA_SIZE) // 2)
    fuse_img[int(img_height / SCALE_FACTOR):int(img_height / SCALE_FACTOR) + CANVA_SIZE,
             bev_x_start:bev_x_start + CANVA_SIZE, :] = bev_view
    
    return fuse_img

def save_final_image(img, token, save_path=VIS_SAVE_PATH):
    """仅保存融合后的最终图像（保持不变）"""
    os.makedirs(save_path, exist_ok=True)
    filename = f"final_vis_{token}.jpg"
    save_full_path = os.path.join(save_path, filename)
    
    cv2.imwrite(save_full_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n✅ 最终融合图像已保存：{save_full_path}")
    print(f"图像尺寸：{img.shape}")
    return save_full_path

# ===================== 4. Pipeline图像处理（完全保持不变） =====================
def mmlabNormalize(img):
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = np.array(img).astype(np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).float()

def sample_augmentation(H, W, is_train=False):
    fH, fW = DATA_CONFIG['input_size']
    if not is_train:
        resize = float(fW) / float(W)
        resize += DATA_CONFIG.get('resize_test', 0.0)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(DATA_CONFIG['crop_h'])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
    return resize, resize_dims, crop, flip, rotate

def img_transform_core(img, resize_dims, crop, flip, rotate):
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    return img

def compute_post_transform(resize, resize_dims, crop, flip, rotate):
    post_rot = torch.eye(2)
    post_tran = torch.zeros(2)
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
    
    post_rot_3x3 = torch.eye(3)
    post_rot_3x3[:2, :2] = post_rot
    post_tran_3x1 = torch.zeros(3)
    post_tran_3x1[:2] = post_tran
    
    return post_rot_3x3, post_tran_3x1

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), -np.sin(h)],
        [np.sin(h), np.cos(h)],
    ])

def optimize_image_processing(front_img):
    imgs = []
    sensor2egos = []
    ego2globals = []
    intrins = []
    post_rots = []
    post_trans = []
    
    src_h, src_w = DATA_CONFIG['src_size']
    input_h, input_w = DATA_CONFIG['input_size']
    
    for cam_name in CAM_NAMES:
        if cam_name == 'CAM_FRONT':
            img_pil = Image.fromarray(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
            original_h, original_w = img_pil.height, img_pil.width
        else:
            img_pil = Image.new('RGB', (src_w, src_h), (0, 0, 0))
            original_h, original_w = src_h, src_w
        
        resize, resize_dims, crop, flip, rotate = sample_augmentation(
            original_h, original_w, is_train=False
        )
        
        img_transformed = img_transform_core(img_pil, resize_dims, crop, flip, rotate)
        img_tensor = mmlabNormalize(img_transformed).unsqueeze(0)
        imgs.append(img_tensor)
        
        post_rot_3x3, post_tran_3x1 = compute_post_transform(resize, resize_dims, crop, flip, rotate)
        post_rots.append(post_rot_3x3.unsqueeze(0))
        post_trans.append(post_tran_3x1.unsqueeze(0))
        
        intrin = get_cam_intrinsic(cam_name).unsqueeze(0)
        intrins.append(intrin)
        
        sensor2ego_info = get_sensor_transforms(cam_name)
        sensor2ego = torch.from_numpy(sensor2ego_info['sensor2ego']).unsqueeze(0)
        
        ego2global = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        sensor2egos.append(sensor2ego)
        ego2globals.append(ego2global)
    
    imgs_tensor = torch.cat(imgs, dim=0).unsqueeze(0).to(DEVICE)
    sensor2egos_tensor = torch.cat(sensor2egos, dim=0).unsqueeze(0).to(DEVICE)
    ego2globals_tensor = torch.cat(ego2globals, dim=0).unsqueeze(0).to(DEVICE)
    intrins_tensor = torch.cat(intrins, dim=0).unsqueeze(0).to(DEVICE)
    post_rots_tensor = torch.cat(post_rots, dim=0).unsqueeze(0).to(DEVICE)
    post_trans_tensor = torch.cat(post_trans, dim=0).unsqueeze(0).to(DEVICE)
    
    bda_mat = torch.eye(4, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    img_inputs = [
        imgs_tensor,        # 0: (1,6,3,256,704)
        sensor2egos_tensor, # 1: (1,6,4,4)
        ego2globals_tensor, # 2: (1,6,4,4)
        intrins_tensor,     # 3: (1,6,3,3)
        post_rots_tensor,   # 4: (1,6,3,3)
        post_trans_tensor,  # 5: (1,6,3)
        bda_mat             # 6: (1,4,4)
    ]
    
    return img_inputs

def load_bevdet_model():
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.data_config = DATA_CONFIG
    print(f"✅ 配置文件加载成功：{CONFIG_PATH}")
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = MMDataParallel(model, device_ids=[0] if DEVICE.startswith('cuda') else [])
    model.eval()
    print(f"✅ 模型加载完成，设备: {DEVICE}")
    return cfg, model

def build_standard_input_data(cfg, img_inputs, lidar_points):
    if lidar_points.shape[1] < 6:
        pad_dim = 6 - lidar_points.shape[1]
        lidar_points = np.pad(lidar_points, ((0,0), (0,pad_dim)), mode='constant')
    lidar_tensor = torch.from_numpy(lidar_points).float().to(DEVICE)
    
    img_metas = [{
        'flip': False,
        'pcd_horizontal_flip': False,
        'pcd_vertical_flip': False,
        'box_mode_3d': Box3DMode.LIDAR,
        'box_type_3d': LiDARInstance3DBoxes,
        'sample_idx': f"zmq_data_{int(time.time())}",
        'pcd_scale_factor': 1.0,
        'img_shape': cfg.data_config['input_size'],
        'ori_shape': cfg.data_config['src_size'],
        'pad_shape': cfg.data_config['input_size'],
        'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        'img_norm_cfg': dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        'cam2img': img_inputs[3].squeeze(0).cpu().numpy(),
        'sensor2ego': img_inputs[1].squeeze(0).cpu().numpy(),
        'ego2global': img_inputs[2].squeeze(0).cpu().numpy(),
        'point_cloud_range': cfg.point_cloud_range if hasattr(cfg, 'point_cloud_range') else [-50, -50, -5, 50, 50, 3],
        'cam_intrinsic': img_inputs[3].squeeze(0).cpu().numpy(),
    }]
    
    input_data = {
        'img_metas': [DataContainer([img_metas], cpu_only=True)],
        'points': [DataContainer([[lidar_tensor]], cpu_only=False)],
        'img_inputs': [img_inputs]
    }
    return input_data

def infer_bevdet(cfg, model, input_data):
    try:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **input_data)
    except Exception as e:
        print(f"❌ 模型推理失败：{str(e)}")
        return [], LiDARInstance3DBoxes(torch.empty(0, 7).to(DEVICE)), torch.empty(0).to(DEVICE), None
    
    if len(result) == 0 or 'pts_bbox' not in result[0]:
        return result, LiDARInstance3DBoxes(torch.empty(0, 7).to(DEVICE)), torch.empty(0).to(DEVICE), None
    
    pts_bbox = result[0]['pts_bbox']
    bbox_key = 'boxes_3d' if 'boxes_3d' in pts_bbox else 'bboxes_3d'
    bboxes_3d = pts_bbox.get(bbox_key, LiDARInstance3DBoxes(torch.empty(0, 7).to(DEVICE)))
    scores = pts_bbox.get('scores_3d', torch.empty(0).to(DEVICE))
    labels = pts_bbox.get('labels_3d', None)
    
    valid_mask = scores >= CONF_THRESH
    valid_bboxes = bboxes_3d[valid_mask] if len(bboxes_3d) > 0 else bboxes_3d
    valid_scores = scores[valid_mask] if len(scores) > 0 else scores
    
    print(f"\n===== 推理结果 =====")
    print(f"原始框数量: {len(bboxes_3d)} | 过滤后: {len(valid_bboxes)}")
    return result, valid_bboxes, valid_scores, labels

# ===================== 5. 【修改】主函数（新增点云缩放参数调参） =====================
def main():
    global VIS_THRED, POINT_RANGE_SCALE
    parser = argparse.ArgumentParser(description='BEVDet可视化 - ZeroMQ数据输入 | 点云绕Z轴旋转90度 | 点云显示缩放 | 仅输出最终融合图像')
    parser.add_argument('--zmq-addr', type=str, default=ZMQ_SUB_ADDR, help='ZeroMQ服务端地址')
    parser.add_argument('--vis-thred', type=float, default=VIS_THRED, help='检测框可视化阈值')
    # 【新增】点云旋转方向参数
    parser.add_argument(
        '--rotate-z', 
        default="counterclockwise", 
        choices=["clockwise", "counterclockwise"],
        help="点云绕Z轴旋转方向（默认：clockwise 顺时针；可选 counterclockwise 逆时针）"
    )
    # 【新增】点云显示范围缩放参数
    parser.add_argument('--point-scale', type=float, default=POINT_RANGE_SCALE, 
                        help='点云显示范围缩放因子（>1扩大显示范围，<1缩小，默认1.0）')
    
    args = parser.parse_args()
    
    # 更新参数
    VIS_THRED = args.vis_thred
    POINT_RANGE_SCALE = args.point_scale  # 更新点云缩放因子
    print(f"📌 当前点云显示范围缩放因子：{POINT_RANGE_SCALE}")
    
    try:
        # 【修改】初始化ZeroMQ接收器（传入点云旋转参数）
        zmq_receiver = ZMQDataReceiver(args.zmq_addr, rotate_z_direction=args.rotate_z)
        
        # 加载模型（保持不变）
        cfg, model = load_bevdet_model()
        
        print("\n===================== 开始接收ZeroMQ数据并处理 =====================")
        print("按 Ctrl+C 终止程序")
        
        # 循环接收数据并处理
        while True:
            try:
                # 【修改】接收ZeroMQ数据（已自动完成点云旋转）
                zmq_data = zmq_receiver.receive_data()
                
                # 图像处理（保持不变）
                img_inputs = optimize_image_processing(zmq_data["image"])
                
                # 构建输入数据（保持不变，使用旋转后的点云）
                input_data = build_standard_input_data(cfg, img_inputs, zmq_data["lidar"])
                
                # 模型推理（保持不变）
                _, valid_bboxes, valid_scores, _ = infer_bevdet(cfg, model, input_data)
                
                # 生成可视化图像（传入点云缩放因子）
                print(f"\n===================== 生成最终可视化图像 =====================")
                vis_imgs = []
                
                for view in VIEWS:
                    if view == 'CAM_FRONT':
                        vis_img = zmq_data["image"].copy()
                        camera_info = get_sensor_transforms(view)
                        if len(valid_bboxes) > 0:
                            vis_img = draw_3d_box_on_img_ref(vis_img, valid_bboxes, camera_info)
                        vis_imgs.append(vis_img)
                    else:
                        vis_imgs.append(np.zeros((900, 1600, 3), dtype=np.uint8))
                
                # 绘制BEV视图（传入点云缩放因子）
                bev_view = draw_bev_view(zmq_data["lidar"], valid_bboxes, valid_scores, point_scale=POINT_RANGE_SCALE)
                
                # 融合图像（保持不变）
                final_img = fuse_img_bev(vis_imgs, bev_view)
                
                # 保存最终图像（保持不变）
                save_final_image(final_img, zmq_data["timestamp"])
                
            except TimeoutError as e:
                print(f"\n⚠️  {e}，继续等待数据...")
                continue
            except Exception as e:
                print(f"\n❌ 单次数据处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
    except KeyboardInterrupt:
        print("\n✅ 用户终止程序")
    except Exception as e:
        print(f"\n❌ 程序初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 【修改】关闭ZeroMQ连接
        if 'zmq_receiver' in locals():
            zmq_receiver.close()
        print("✅ 程序正常退出")

if __name__ == "__main__":
    main()
