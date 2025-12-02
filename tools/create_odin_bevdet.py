import os
import pickle
import numpy as np
from datetime import datetime

# -------------------------- 配置参数（关键：路径直接指向odin目录）--------------------------
ACTUAL_DATA_ROOT = "/workspace/BEV/BEVDet/data/nuscenes/odin"  # 你的数据根目录
OUTPUT_PKL = "/workspace/BEV/BEVDet/data/nuscenes/odin_infos_test.pkl"
TARGET_CAMS = [
    'CAM_FRONT', 
    'CAM_FRONT_RIGHT', 
    'CAM_FRONT_LEFT', 
    'CAM_BACK', 
    'CAM_BACK_LEFT', 
    'CAM_BACK_RIGHT'
]
LIDAR_DIR = "LIDAR_TOP"
CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# -------------------------- 6相机标定参数（保持nuScenes标准，仅修改CAM_FRONT）--------------------------
CAM_PARAMS = {
    # 'CAM_FRONT': {
    #     "sensor2ego_translation": [1.70079118954, 0.0159456324149, 1.51095763913],  # 保持原ego相对位置（若有实际值可修改）
    #     "sensor2ego_rotation": [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],  # 保持原ego旋转（若有实际值可修改）
    #     # ========== 修改：CAM_FRONT到LiDAR的外参（从Tcl_0 4x4齐次矩阵拆分） ==========
    #     "sensor2lidar_translation": [0.02344, -0.0049, -0.03012],  # 取自Tcl_0第4列前3个元素（x,y,z）
    #     "sensor2lidar_rotation": np.array([
    #         [0.00095, -0.99997, -0.00735],
    #         [-0.00365, 0.00734, -0.99997],
    #         [0.99999, 0.00098, -0.00364]
    #     ], dtype=np.float64),  # 取自Tcl_0前3x3旋转矩阵
    #     # ========== 修改：CAM_FRONT内参（鱼眼相机FishPoly参数构建） ==========
    #     "cam_intrinsic": np.array([
    #         [1266.41720,0.0, 816.267020],  # A11, A12, u0
    #         [0.0, 1266.41720, 491.507066],  # 0, A22, v0
    #         [0.0, 0.0, 1.0]
    #     ], dtype=np.float64)  # 内参矩阵格式：[A11,A12,u0; 0,A22,v0; 0,0,1]
    # },
    'CAM_FRONT': {
        "sensor2ego_translation": [1.70079118954, 0.0159456324149, 1.51095763913],
        "sensor2ego_rotation": [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
        "sensor2lidar_translation": [-0.01271581, 0.76880558, -0.31059456],
        "sensor2lidar_rotation": np.array([
            [0.99996937, 0.0067556, -0.0039516],
            [0.00382456, 0.01871645, 0.99981752],
            [0.00682833, -0.99980201, 0.01869004]
        ], dtype=np.float64),
          "cam_intrinsic": np.array([
            [1266.41720,0.0, 816.267020],  # A11, A12, u0
            [0.0, 1266.41720, 491.507066],  # 0, A22, v0
            [0.0, 0.0, 1.0]
        ], dtype=np.float64) 
    },
    'CAM_FRONT_RIGHT': {
        "sensor2ego_translation": [1.5508477543, -0.493404796419, 1.49574800619],
        "sensor2ego_rotation": [0.2060347966337182, -0.2026940577919598, 0.6824507824531167, -0.6713610884174485],
        "sensor2lidar_translation": [0.49650027, 0.61746215, -0.32655959],
        "sensor2lidar_rotation": np.array([
            [0.5518728, -0.01045233, 0.83386279],
            [-0.83352138, 0.02431967, 0.55195169],
            [-0.02604845, -0.99964959, 0.00470913]
        ], dtype=np.float64),
        "cam_intrinsic": np.array([
            [1260.84744, 0.0, 807.968245],
            [0.0, 1260.84744, 495.334427],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    },
    'CAM_FRONT_LEFT': {
        "sensor2ego_translation": [1.52387798135, 0.494631336551, 1.50932822144],
        "sensor2ego_rotation": [0.6757265034669446, -0.6736266522251881, 0.21214015046209478, -0.21122827103904068],
        "sensor2lidar_translation": [-0.4917212, 0.59365311, -0.31925387],
        "sensor2lidar_rotation": np.array([
            [0.5726028, 0.00270925, -0.81982845],
            [0.81955664, 0.02406756, 0.5724925],
            [0.0212823, -0.99970666, 0.01156077]
        ], dtype=np.float64),
        "cam_intrinsic": np.array([
            [1272.59795, 0.0, 826.615493],
            [0.0, 1272.59795, 479.751654],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    },
    'CAM_BACK': {
        "sensor2ego_translation": [0.0283260309358, 0.00345136761476, 1.57910346144],
        "sensor2ego_rotation": [0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578],
        "sensor2lidar_translation": [-0.00369546, -0.90757475, -0.28322187],
        "sensor2lidar_rotation": np.array([
            [-0.99994132, 0.00984665, -0.00451724],
            [0.00459128, 0.00750977, -0.99996126],
            [-0.00981234, -0.99992332, -0.00755454]
        ], dtype=np.float64),
        "cam_intrinsic": np.array([
            [809.22099, 0.0, 829.21960],
            [0.0, 809.22099, 481.77842],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    },
    'CAM_BACK_LEFT': {
        "sensor2ego_translation": [1.03569100218, 0.484795032713, 1.59097014818],
        "sensor2ego_rotation": [0.6924185592174665, -0.7031619420114925, -0.11648342771943819, 0.11203317912370753],
        "sensor2lidar_translation": [-0.48313021, 0.09925075, -0.24976868],
        "sensor2lidar_rotation": np.array([
            [-0.3170637, 0.01989561, -0.94819553],
            [0.94807792, 0.03287372, -0.3163346],
            [0.02487704, -0.99926147, -0.02928565]
        ], dtype=np.float64),
        "cam_intrinsic": np.array([
            [1256.74148, 0.0, 792.11257],
            [0.0, 1256.74148, 492.77575],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    },
    'CAM_BACK_RIGHT': {
        "sensor2ego_translation": [1.0148780988, -0.480568219723, 1.56239545128],
        "sensor2ego_rotation": [0.12280980120078765, -0.132400842670559, -0.7004305821388234, 0.690496031265798],
        "sensor2lidar_translation": [0.48231268, 0.07918378, -0.2730718],
        "sensor2lidar_rotation": np.array([
            [-0.35672464, -0.0054139, 0.93419389],
            [-0.93353071, 0.04018099, -0.35623855],
            [-0.0356082, -0.99917775, -0.01938759]
        ], dtype=np.float64),
        "cam_intrinsic": np.array([
            [1259.51374, 0.0, 807.25291],
            [0.0, 1259.51374, 501.19580],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    }
}

# ========== 修改：LiDAR到自车标定参数（同一坐标系，无平移旋转） ==========
# LIDAR2EGO = {
#     "translation": [0.0, 0.0, 0.0],  # 无平移
#     "rotation": [1.0, 0.0, 0.0, 0.0]  # 无旋转（单位四元数：w=1, x=0, y=0, z=0）
# }
LIDAR2EGO = {
    "translation": [0.943713, 0.0, 1.84023],
    "rotation": [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
}

EGO2GLOBAL = {
    "translation": [249.89610931430778, 917.5522573162784, 0.0],
    "rotation": [0.9984303573176436, -0.008635865272570774, 0.0025833156025800875, -0.05527720957189669]
}

# -------------------------- 工具函数（保持不变）--------------------------
def get_file_timestamp(filename):
    parts = filename.split('_')
    if len(parts) >= 4:
        ts_str = parts[-1].split('.')[0]
        if ts_str.isdigit() and len(ts_str) == 16:
            return int(ts_str)
    return int(datetime.now().timestamp() * 1e6)

def extract_frame_id(filename):
    parts = filename.split('_')
    if len(parts) >= 4:
        frame_str = parts[-1].split('.')[0]
        if frame_str.isdigit() and len(frame_str) == 16:
            return frame_str
    raise ValueError(f"文件名格式错误：{filename}（应为：nxxx-xxxx-xxxx-xxxx_XXX_CAM_XXX_16位时间戳.jpg）")

def generate_nuscenes_style_pkl():
    # 1. 验证目录存在
    cam_dirs = {cam: os.path.join(ACTUAL_DATA_ROOT, cam) for cam in TARGET_CAMS}
    lidar_dir = os.path.join(ACTUAL_DATA_ROOT, LIDAR_DIR)
    for cam, dir_path in cam_dirs.items():
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"相机目录不存在：{dir_path}")
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"LiDAR目录不存在：{lidar_dir}")

    # 2. 收集相机文件
    cam_frame_maps = {}
    for cam in TARGET_CAMS:
        cam_dir = cam_dirs[cam]
        cam_files = sorted([
            f for f in os.listdir(cam_dir) 
            if f.endswith('.jpg') and f.startswith('n') and cam in f
        ])
        if not cam_files:
            raise ValueError(f"未找到{cam}的图像文件：{cam_dir}")
        frame_map = {extract_frame_id(f): f for f in cam_files}
        cam_frame_maps[cam] = frame_map
        print(f"找到 {len(cam_files)} 张{cam}图像")

    # 3. 收集LiDAR文件
    lidar_files = sorted([
        f for f in os.listdir(lidar_dir) 
        if f.endswith('.pcd.bin') and f.startswith('n') and LIDAR_DIR in f
    ])
    if not lidar_files:
        raise ValueError(f"未找到LiDAR文件：{lidar_dir}")
    lidar_frame_map = {extract_frame_id(f): f for f in lidar_files}
    print(f"找到 {len(lidar_files)} 帧LiDAR点云")

    # 4. 匹配同步帧
    common_frames = set(cam_frame_maps[TARGET_CAMS[0]].keys())
    for cam in TARGET_CAMS[1:]:
        common_frames.intersection_update(cam_frame_maps[cam].keys())
    common_frames.intersection_update(lidar_frame_map.keys())
    common_frames = sorted(list(common_frames))
    if not common_frames:
        raise ValueError("无同步帧！")
    print(f"匹配到 {len(common_frames)} 对同步帧")

    # 5. 构建样本信息（核心修改：路径直接指向 odin 目录）
    infos = []
    for idx, frame_id in enumerate(common_frames):
        cams_info = {}
        for cam in TARGET_CAMS:
            cam_fn = cam_frame_maps[cam][frame_id]
            # 关键修改：路径格式改为 ./data/nuscenes/odin/CAM_FRONT/xxx.jpg（与实际存储一致）
            cam_path = os.path.join("./data/nuscenes/odin", cam, cam_fn)
            # 验证实际路径
            actual_cam_path = os.path.join("/workspace/BEV/BEVDet", cam_path)
            if not os.path.exists(actual_cam_path):
                raise FileNotFoundError(f"{cam}路径无效：{actual_cam_path}")
            
            cam_params = CAM_PARAMS[cam]
            cams_info[cam] = {
                "data_path": cam_path,  # 存储真实路径（odin目录下）
                "type": cam,
                "sample_data_token": f"{frame_id}_{cam}",
                "sensor2ego_translation": cam_params["sensor2ego_translation"],
                "sensor2ego_rotation": cam_params["sensor2ego_rotation"],
                "ego2global_translation": EGO2GLOBAL["translation"],
                "ego2global_rotation": EGO2GLOBAL["rotation"],
                "timestamp": get_file_timestamp(cam_fn),
                "sensor2lidar_translation": cam_params["sensor2lidar_translation"],
                "sensor2lidar_rotation": cam_params["sensor2lidar_rotation"],
                "cam_intrinsic": cam_params["cam_intrinsic"]
            }

        # LiDAR路径同样指向 odin 目录
        lidar_fn = lidar_frame_map[frame_id]
        lidar_path = os.path.join("./data/nuscenes/odin", LIDAR_DIR, lidar_fn)
        actual_lidar_path = os.path.join("/workspace/BEV/BEVDet", lidar_path)
        if not os.path.exists(actual_lidar_path):
            raise FileNotFoundError(f"LiDAR路径无效：{actual_lidar_path}")

        # 构建样本信息
        sample_token = f"{frame_id}_LIDAR_TOP"
        sample_ts = max([cams_info[cam]["timestamp"] for cam in TARGET_CAMS] + [get_file_timestamp(lidar_fn)])

        gt_boxes = np.array([], dtype=np.float64).reshape(0, 7)
        gt_names = np.array([], dtype=np.str_)
        gt_velocity = np.array([], dtype=np.float64).reshape(0, 2)
        num_lidar_pts = np.array([], dtype=np.int64)
        num_radar_pts = np.array([], dtype=np.int64)
        valid_flag = np.array([], dtype=bool)
        ann_infos = ([], [])

        sample_info = {
            "lidar_path": lidar_path,
            "token": sample_token,
            "sweeps": [],
            "cams": cams_info,
            "lidar2ego_translation": LIDAR2EGO["translation"],  # 使用修改后的LiDAR-自车平移
            "lidar2ego_rotation": LIDAR2EGO["rotation"],  # 使用修改后的LiDAR-自车旋转
            "ego2global_translation": EGO2GLOBAL["translation"],
            "ego2global_rotation": EGO2GLOBAL["rotation"],
            "timestamp": sample_ts,
            "gt_boxes": gt_boxes,
            "gt_names": gt_names,
            "gt_velocity": gt_velocity,
            "num_lidar_pts": num_lidar_pts,
            "num_radar_pts": num_radar_pts,
            "valid_flag": valid_flag,
            "sample_token": sample_token,
            "ego_pose_token": f"ego_pose_{sample_token}",
            "calibrated_sensor_token": f"calib_{sample_token}",
            "scene_token": "scene_0000",
            "frame_id": frame_id,
            "prev": f"sample_{common_frames[idx-1]}" if idx > 0 else "",
            "next": f"sample_{common_frames[idx+1]}" if idx < len(common_frames)-1 else "",
            "ann_infos": ann_infos,
            "occ_path": ""
        }

        infos.append(sample_info)
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(common_frames)} 帧")

    # 6. 保存pkl
    pkl_data = {
        "infos": infos,
        "metadata": {
            "version": "v1.0-trainval",
            "dataset_name": "nuScenes",
            "class_names": CLASSES,
            "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "cam_names": TARGET_CAMS
        }
    }

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(pkl_data, f)
    print(f"\n✅ pkl生成完成：{OUTPUT_PKL}")

if __name__ == "__main__":
    try:
        generate_nuscenes_style_pkl()
    except Exception as e:
        print(f"❌ 生成失败：{e}")
        raise