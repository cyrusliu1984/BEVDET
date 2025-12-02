import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import numpy as np
import zmq
import pickle
import cv2

# 配置
IMAGE_TOPIC = "/odin1/image"
LIDAR_TOPIC = "/odin1/cloud_raw"
ZMQ_ADDR = "tcp://127.0.0.1:5555"  # 通信地址（本地回环）
NUSCENES_IMG_SIZE = (1600, 900)
ROTATE_Z_DIRECTION = "clockwise"


class ROS2DataForwarder(Node):
    def __init__(self):
        super().__init__("ros2_data_forwarder")
        
        # 初始化ROS2工具
        self.bridge = CvBridge()
        self.image_queue = []  # (timestamp_ns, cv_img)
        self.lidar_queue = []  # (timestamp_ns, points_np)
        self.queue_max_size = 20
        self.forwarded_frame_count = 0  # ←←← 新增：全局转发帧计数器

        # ZeroMQ 初始化
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.PUB)
        self.socket.bind(ZMQ_ADDR)
        self.get_logger().info(f"ZeroMQ绑定成功：{ZMQ_ADDR}")

        # ROS2 QoS配置
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 订阅话题
        self.image_sub = self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, qos_profile)
        self.lidar_sub = self.create_subscription(PointCloud2, LIDAR_TOPIC, self.lidar_callback, qos_profile)

        # 定时器：每100ms尝试同步并转发
        self.timer = self.create_timer(0.1, self.forward_sync_data)

    def rotate_point_cloud_z(self, points_np):
        """绕Z轴旋转点云"""
        x = points_np[:, 0]
        y = points_np[:, 1]
        z = points_np[:, 2]
        intensity = points_np[:, 3]
        if ROTATE_Z_DIRECTION == "clockwise":
            new_x = y
            new_y = -x
        else:
            new_x = -y
            new_y = x
        return np.column_stack([new_x, new_y, z, intensity]).astype(np.float32)

    def image_callback(self, msg):
        """接收图像并缓存"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            cv_img_resized = cv2.resize(cv_img, NUSCENES_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            self.image_queue.append((timestamp, cv_img_resized))
            if len(self.image_queue) > self.queue_max_size:
                self.image_queue.pop(0)
        except Exception as e:
            self.get_logger().error(f"图像回调失败：{e}")

    def lidar_callback(self, msg):
        """接收点云并缓存"""
        try:
            points_generator = pc2.read_points(
                msg, 
                field_names=["x", "y", "z", "intensity"], 
                skip_nans=True
            )
            points_list = list(points_generator)
            if len(points_list) == 0:
                self.get_logger().warn("点云为空，跳过")
                return

            x = np.array([p[0] for p in points_list], dtype=np.float32)
            y = np.array([p[1] for p in points_list], dtype=np.float32)
            z = np.array([p[2] for p in points_list], dtype=np.float32)
            intensity = np.array([p[3] for p in points_list], dtype=np.uint8).astype(np.float32) / 255.0

            points_np = np.column_stack([x, y, z, intensity])
            points_rotated = self.rotate_point_cloud_z(points_np)

            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            self.lidar_queue.append((timestamp, points_rotated))
            if len(self.lidar_queue) > self.queue_max_size:
                self.lidar_queue.pop(0)
        except Exception as e:
            self.get_logger().error(f"点云回调失败：{e}")

    def forward_sync_data(self):
        """时间同步后转发数据"""
        if not self.image_queue or not self.lidar_queue:
            return

        # 排序确保时间顺序
        self.image_queue.sort(key=lambda x: x[0])
        self.lidar_queue.sort(key=lambda x: x[0])
        lidar_timestamps = [t for t, _ in self.lidar_queue]
        matched = []

        for img_ts, img in self.image_queue[:]:  # 使用副本避免修改中遍历
            idx = np.searchsorted(lidar_timestamps, img_ts)
            candidates = []
            if idx < len(lidar_timestamps):
                candidates.append((lidar_timestamps[idx], idx))
            if idx > 0:
                candidates.append((lidar_timestamps[idx - 1], idx - 1))
            if not candidates:
                continue

            closest_ts, closest_idx = min(candidates, key=lambda x: abs(x[0] - img_ts))
            if abs(closest_ts - img_ts) <= 1e8:  # 100ms 同步窗口
                lidar_data = self.lidar_queue[closest_idx][1]
                matched.append((img_ts, img, lidar_data))
                # 从队列中移除已匹配项
                del self.lidar_queue[closest_idx]
                lidar_timestamps.pop(closest_idx)

        # 转发匹配的数据
        for img_ts, img, lidar in matched:
            data = {
                "timestamp": img_ts,
                "image": img,
                "lidar": lidar
            }
            serialized_data = pickle.dumps(data)
            self.socket.send(serialized_data)
            self.forwarded_frame_count += 1  # ←←← 关键：递增全局计数
            self.get_logger().info(
                f"转发第 {self.forwarded_frame_count} 帧数据（点云点数：{lidar.shape[0]}）"
            )

        # 清除已转发的图像
        if matched:
            # 只保留未匹配的图像（按时间戳过滤）
            matched_img_ts = {img_ts for img_ts, _, _ in matched}
            self.image_queue = [(ts, img) for ts, img in self.image_queue if ts not in matched_img_ts]


def main():
    rclpy.init()
    forwarder = ROS2DataForwarder()
    try:
        rclpy.spin(forwarder)
    except KeyboardInterrupt:
        pass
    finally:
        forwarder.socket.close()
        forwarder.zmq_context.term()
        forwarder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()