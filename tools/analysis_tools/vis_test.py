import json
import pickle
import os
import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB

# -------------------------- 配置参数（必须根据你的路径修改）--------------------------
JSON_PATH = "/workspace/BEV/BEVDet/save_path/pts_bbox/results_nusc.json"
PKL_PATH = "/workspace/BEV/BEVDet/data/nuscenes/odin_infos_test.pkl"
SAVE_DIR = "/workspace/BEV/BEVDet/vis"
CONF_THRESH = 0.3
SELECT_FRAMES = [200, 300]  # 只看有检测结果的帧
CAM_VIEW = "CAM_FRONT"

CLASS_COLOR = {
    'car': (0, 255, 0), 'truck': (255, 165, 0), 'pedestrian': (0, 0, 255),
    'bicycle': (255, 0, 255), 'bus': (0, 255, 255), 'trailer': (128, 0, 128),
    'construction_vehicle': (192, 192, 192), 'barrier': (255, 255, 0),
    'motorcycle': (144, 238, 144), 'traffic_cone': (255, 192, 203)
}

# -------------------------- 修正后的坐标转换函数 --------------------------
def lidar2img(points_lidar, camera_info):
    points_lidar_homogeneous = np.concatenate([
        points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)
    ], axis=1)
    
    # 修正：激光雷达→相机的外参计算（关键！）
    R_cam2lidar = camera_info['sensor2lidar_rotation'].astype(np.float32)
    t_cam2lidar = np.array(camera_info['sensor2lidar_translation'], dtype=np.float32)
    R_lidar2cam = R_cam2lidar.T  # 旋转矩阵转置（逆）
    t_lidar2cam = -R_lidar2cam @ t_cam2lidar  # 平移向量修正
    
    # 构建转换矩阵
    lidar2camera = np.eye(4, dtype=np.float32)
    lidar2camera[:3, :3] = R_lidar2cam
    lidar2camera[:3, 3] = t_lidar2cam
    
    # 转换到相机坐标系
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    
    # 过滤无效点（深度>0.5米）
    valid = points_camera[:, 2] > 0.5
    points_camera_valid = points_camera[valid]
    if len(points_camera_valid) == 0:
        return np.array([]), valid
    
    # 透视投影
    points_camera_proj = points_camera_valid / points_camera_valid[:, 2:3]
    
    # 内参投影到像素
    K = camera_info['cam_intrinsic'].astype(np.float32)
    points_img = points_camera_proj @ K.T
    points_img = points_img[:, :2].astype(np.int32)
    
    return points_img, valid

# -------------------------- 绘制函数 --------------------------
def draw_3d_box_on_img(img, corners_img, class_name, score):
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    color = CLASS_COLOR.get(class_name, (255, 255, 255))
    
    # 绘制有效边（只绘制两点都在图像内的边）
    img_h, img_w = img.shape[:2]
    for (p1, p2) in box_edges:
        x1, y1 = corners_img[p1]
        x2, y2 = corners_img[p2]
        # 检查点是否在图像内
        if 0 <= x1 < img_w and 0 <= y1 < img_h and 0 <= x2 < img_w and 0 <= y2 < img_h:
            cv2.line(img, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签（选一个在图像内的角点）
    for i, (x, y) in enumerate(corners_img):
        if 0 <= x < img_w and 0 <= y < img_h:
            label = f"{class_name} {score:.2f}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            break

# -------------------------- 主逻辑 --------------------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    with open(JSON_PATH, 'r') as f:
        det_results = json.load(f)['results']
    with open(PKL_PATH, 'rb') as f:
        dataset = pickle.load(f)['infos']
    
    # 逐帧处理
    for frame_idx in SELECT_FRAMES:
        frame_info = dataset[frame_idx]
        sample_token = frame_info['token']
        cam_info = frame_info['cams'][CAM_VIEW]
        
        # 读取图像
        img_path = cam_info['data_path']
        if not os.path.exists(img_path):
            img_path = os.path.join("/workspace/BEV/BEVDet", img_path)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        print(f"\n处理帧{frame_idx}：图像尺寸 {img_w}x{img_h}")
        
        # 检查检测结果
        if sample_token not in det_results:
            print(f"帧{frame_idx}无检测结果")
            continue
        detections = [d for d in det_results[sample_token] if d['detection_score'] >= CONF_THRESH]
        if not detections:
            print(f"帧{frame_idx}无置信度≥{CONF_THRESH}的检测结果")
            cv2.putText(img, "No Valid Detections", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{frame_idx}_no_det.jpg"), img)
            continue
        
        # 转换检测框为3D角点
        det_boxes = []
        det_classes = []
        det_scores = []
        for det in detections:
            translation = det['translation']
            size = det['size']
            # 修正：yaw角不额外加π/2（根据实际情况调整）
            yaw = Quaternion(det['rotation']).yaw_pitch_roll[0]
            det_boxes.append(translation + size + [yaw])
            det_classes.append(det['detection_name'])
            det_scores.append(det['detection_score'])
        
        # 生成8个角点
        det_boxes = np.array(det_boxes, dtype=np.float32)
        lidar_boxes = LB(det_boxes, origin=(0.5, 0.5, 0.0))
        corners_lidar = lidar_boxes.corners.numpy()  # (N,8,3)
        
        # 投影并绘制
        for i, (corners_3d, cls, score) in enumerate(zip(corners_lidar, det_classes, det_scores)):
            print(f"  检测目标{i}：{cls}，置信度{score:.2f}，3D角点范围：x[{corners_3d[:,0].min():.1f},{corners_3d[:,0].max():.1f}]")
            corners_img, valid = lidar2img(corners_3d, cam_info)
            if len(corners_img) == 0:
                print(f"  目标{i}投影后无有效点，跳过")
                continue
            
            # 绘制3D框
            draw_3d_box_on_img(img, corners_img, cls, score)
        
        # 保存图像
        save_path = os.path.join(SAVE_DIR, f"frame_{frame_idx}_fixed.jpg")
        cv2.imwrite(save_path, img)
        print(f"已保存：{save_path}")
    
    print(f"\n✅ 处理完成，结果保存在：{SAVE_DIR}")