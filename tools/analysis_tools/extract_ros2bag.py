import os
import argparse
import numpy as np
import cv2
from datetime import datetime
from cv_bridge import CvBridge
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import bisect  # 用于时间戳匹配


def rotate_point_cloud_z(points_np, direction="clockwise"):
    """
    对点云绕 Z 轴旋转 90 度
    :param points_np: 原始点云数组，shape=(N,4)，格式为 [x, y, z, intensity]
    :param direction: 旋转方向，可选 "clockwise"（顺时针）或 "counterclockwise"（逆时针）
    :return: 旋转后的点云数组
    """
    # 提取 x, y 坐标（Z 坐标和强度不变）
    x = points_np[:, 0]
    y = points_np[:, 1]
    z = points_np[:, 2]
    intensity = points_np[:, 3]

    # 绕 Z 轴旋转 90 度（右手坐标系：X向右，Y向前，Z向上）
    if direction == "clockwise":
        # 顺时针旋转 90 度：x' = y, y' = -x
        new_x = y
        new_y = -x
    elif direction == "counterclockwise":
        # 逆时针旋转 90 度：x' = -y, y' = x
        new_x = -y
        new_y = x
    else:
        raise ValueError("direction 只能是 'clockwise' 或 'counterclockwise'")

    # 重组点云（保持 [x, y, z, intensity] 格式）
    rotated_points = np.column_stack([new_x, new_y, z, intensity])
    return rotated_points.astype(np.float32)


def extract_bag(bag_path, output_dir, prefix="n008", rotate_z_direction="clockwise"):
    """从ROS2 bag包中提取点云和图像话题，点云绕Z轴旋转90度，图像缩放到1600x900（nuScenes标准）"""
    # 定义所有需要的相机目录
    cam_front = "CAM_FRONT"
    other_cams = [
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT"
    ]
    all_cams = [cam_front] + other_cams
    lidar_dir_name = "LIDAR_TOP"

    # nuScenes 标准图像尺寸（宽x高）
    NUSCENES_IMG_SIZE = (1600, 900)  # 关键配置：统一缩放到该尺寸
    print(f"目标图像尺寸：{NUSCENES_IMG_SIZE[0]}x{NUSCENES_IMG_SIZE[1]}（nuScenes标准）")

    # 创建所有输出目录
    for cam in all_cams:
        cam_dir = os.path.join(output_dir, cam)
        os.makedirs(cam_dir, exist_ok=True)
    lidar_dir = os.path.join(output_dir, lidar_dir_name)
    os.makedirs(lidar_dir, exist_ok=True)

    print(f"输出目录：{output_dir}")
    print(f"图像目录：{[os.path.join(output_dir, cam) for cam in all_cams]}")
    print(f"点云目录：{lidar_dir}")
    print(f"点云绕Z轴旋转方向：{rotate_z_direction} 90度")

    # 初始化ROS2和cv_bridge
    rclpy.init()
    bridge = CvBridge()

    # 配置bag读取器
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 获取bag中的所有话题及类型
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    print(f"bag包中包含的话题：{list(type_map.keys())}")

    # 目标话题（根据实际bag配置）
    target_topics = {
        "/odin1/image/undistorted": "image",
        "/odin1/cloud_raw": "lidar"
    }

    # 缓存图像和点云消息（时间戳+消息数据）
    images = []  # 元素: (timestamp, Image_msg)
    lidars = []  # 元素: (timestamp, PointCloud2_msg)

    # 第一步：读取并缓存所有目标话题消息
    print("正在读取bag包数据...")
    while reader.has_next():
        topic, data, timestamp = reader.read_next()  # timestamp单位：纳秒
        if topic not in target_topics:
            continue

        # 缓存图像消息
        if topic == "/odin1/image/undistorted" and type_map[topic] == "sensor_msgs/msg/Image":
            msg = deserialize_message(data, Image)
            images.append((timestamp, msg))
        
        # 缓存点云消息
        elif topic == "/odin1/cloud_raw" and type_map[topic] == "sensor_msgs/msg/PointCloud2":
            msg = deserialize_message(data, PointCloud2)
            lidars.append((timestamp, msg))

    # 按时间戳排序（确保后续匹配正确）
    images.sort(key=lambda x: x[0])
    lidars.sort(key=lambda x: x[0])
    print(f"原始数据：图像{len(images)}张，点云{len(lidars)}帧")

    # 第二步：时间同步匹配（只保留时间差<100ms的成对数据）
    matched_pairs = []  # 元素: (img_ts, img_msg, lidar_ts, lidar_msg)
    lidar_timestamps = [t for t, _ in lidars]  # 点云时间戳列表（用于快速匹配）
    time_threshold = 1e8  # 时间差阈值（100ms，单位：纳秒）

    for img_ts, img_msg in images:
        # 找到最接近图像时间戳的点云
        idx = bisect.bisect_left(lidar_timestamps, img_ts)
        candidates = []
        # 检查左右相邻的点云时间戳
        if idx < len(lidar_timestamps):
            candidates.append((lidar_timestamps[idx], idx))
        if idx > 0:
            candidates.append((lidar_timestamps[idx-1], idx-1))
        if not candidates:
            continue  # 无匹配点云
        
        # 选择时间差最小的点云
        closest_ts, closest_idx = min(candidates, key=lambda x: abs(x[0] - img_ts))
        if abs(closest_ts - img_ts) <= time_threshold:
            # 匹配成功，加入成对列表，并从点云列表中移除（避免重复匹配）
            matched_pairs.append((img_ts, img_msg, closest_ts, lidars[closest_idx][1]))
            del lidars[closest_idx]
            del lidar_timestamps[closest_idx]

    print(f"时间匹配后：共{len(matched_pairs)}对（图像+点云）")
    if len(matched_pairs) == 0:
        print("警告：未找到匹配的图像和点云数据！")
        rclpy.shutdown()
        return

    # 第三步：处理匹配对，生成文件
    count = 0  # 统一计数（确保图像和点云序号一致）

    for img_ts, img_msg, lidar_ts, lidar_msg in matched_pairs:
        # -------------------------- 处理CAM_FRONT图像（缩放到1600x900） --------------------------
        try:
            # 转换图像格式
            cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
            # 单通道转BGR（确保可保存为JPG）
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            
            # 关键步骤：缩放到nuScenes标准尺寸（1600x900）
            # 使用INTER_LINEAR插值（适合缩小图像，画质更清晰）
            cv_img_resized = cv2.resize(cv_img, NUSCENES_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            print(f"图像缩放完成：原始尺寸{cv_img.shape[1]}x{cv_img.shape[0]} → 目标尺寸{cv_img_resized.shape[1]}x{cv_img_resized.shape[0]}")

            # 生成图像文件名（统一用图像时间戳）
            dt = datetime.fromtimestamp(img_ts / 1e9)
            datetime_str = dt.strftime("%Y%m%d-%H%M%S")
            timestamp_us = img_ts // 1000  # 纳秒转微秒
            img_filename = f"{prefix}-{datetime_str}-{count:04d}_{cam_front}_{timestamp_us}.jpg"
            img_path = os.path.join(output_dir, cam_front, img_filename)
            cv2.imwrite(img_path, cv_img_resized)  # 保存缩放后的图像

        except Exception as e:
            print(f"图像处理失败（跳过该对）：{e}")
            continue

        # -------------------------- 生成其他相机黑色图像（直接用1600x900尺寸） --------------------------
        # 直接创建nuScenes标准尺寸的黑色图像，无需跟随原始图像尺寸
        black_img = np.zeros((NUSCENES_IMG_SIZE[1], NUSCENES_IMG_SIZE[0], 3), dtype=np.uint8)
        for cam in other_cams:
            black_filename = f"{prefix}-{datetime_str}-{count:04d}_{cam}_{timestamp_us}.jpg"
            black_path = os.path.join(output_dir, cam, black_filename)
            cv2.imwrite(black_path, black_img)

        # -------------------------- 处理点云（含绕Z轴旋转90度） --------------------------
        try:
            # 解析点云（x,y,z,intensity）
            points = pc2.read_points(lidar_msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
            points_struct = np.array(list(points))
            # 转换为float32数组（N,4）
            x = points_struct['x'].astype(np.float32)
            y = points_struct['y'].astype(np.float32)
            z = points_struct['z'].astype(np.float32)
            intensity = points_struct['intensity'].astype(np.float32)
            points_np = np.column_stack([x, y, z, intensity])

            # 关键步骤：点云绕Z轴旋转90度
            points_np_rotated = rotate_point_cloud_z(points_np, direction=rotate_z_direction)

            # 生成点云文件名（与图像序号一致）
            lidar_filename = f"{prefix}-{datetime_str}-{count:04d}_{lidar_dir_name}_{timestamp_us}.pcd.bin"
            lidar_path = os.path.join(lidar_dir, lidar_filename)
            points_np_rotated.tofile(lidar_path)  # 保存旋转后的点云

        except Exception as e:
            print(f"点云处理失败（跳过该对）：{e}")
            continue

        # 进度提示
        count += 1
        if count % 10 == 0:
            print(f"已处理 {count}/{len(matched_pairs)} 对数据")

    print(f"提取完成！共生成：")
    print(f"- {cam_front}图像（1600x900）：{count}张")
    for cam in other_cams:
        print(f"- {cam}黑色图像（1600x900）：{count}张")
    print(f"- {lidar_dir_name}点云（已绕Z轴旋转90度）：{count}帧")

    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从ROS2 bag包中提取图像和点云（图像缩放到1600x900，点云绕Z轴旋转90度）")
    parser.add_argument("bag_path", help="ROS2 bag包路径（如 ./my_recording）")
    parser.add_argument("output_dir", help="输出目录（如 ./extracted_data）")
    parser.add_argument("--prefix", default="n008", help="文件名前缀（默认：n008）")
    parser.add_argument(
        "--rotate-z", 
        default="clockwise", 
        choices=["clockwise", "counterclockwise"],
        help="点云绕Z轴旋转方向（默认：clockwise 顺时针；可选 counterclockwise 逆时针）"
    )
    args = parser.parse_args()

    if not os.path.exists(args.bag_path):
        print(f"错误：bag包路径不存在 - {args.bag_path}")
        exit(1)

    extract_bag(args.bag_path, args.output_dir, prefix=args.prefix, rotate_z_direction=args.rotate_z)