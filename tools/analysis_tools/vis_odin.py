# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
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
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result in json format')
    parser.add_argument(
        '--show-range',
        type=int,
        default=50,
        help='Range of visualization in BEV')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--vis-thred',
        type=float,
        default=0.3,
        help='Threshold the predicted results')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='video',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args


color_map = {0: (255, 255, 0), 1: (0, 255, 255)}


def main():
    args = parse_args()
    res = json.load(open(args.res, 'r'))
    info_path = args.root_path + '/odin_infos_test.pkl'
    dataset = pickle.load(open(info_path, 'rb'))
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size  # BEV画布尺寸（1000x1000）
    show_range = args.show_range
    target_img_height = 900  # 统一图像高度

    vout = None
    if args.format == 'video':
        pass

    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]
    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')
    for cnt, infos in enumerate(
            dataset['infos'][:min(args.vis_frames, len(dataset['infos']))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))
        
        # 收集实例（略）
        pred_res = res['results'][infos['token']]
        pred_boxes = [
            pred_res[rid]['translation'] + pred_res[rid]['size'] + [
                Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0] +
                np.pi / 2
            ] for rid in range(len(pred_res))
        ]
        if len(pred_boxes) == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_boxes = np.array(pred_boxes, dtype=np.float32)
            boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
            corners_global = boxes.corners.numpy().reshape(-1, 3)
            corners_global = np.concatenate(
                [corners_global, np.ones([corners_global.shape[0], 1])], axis=1)
            l2g = get_lidar2global(infos)
            corners_lidar = corners_global @ np.linalg.inv(l2g).T
            corners_lidar = corners_lidar[:, :3]
        pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=np.bool_)
        scores = [pred_res[rid]['detection_score'] for rid in range(len(pred_res))]
        if args.draw_gt:
            gt_boxes = infos['gt_boxes']
            gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
            width = gt_boxes[:, 4].copy()
            gt_boxes[:, 4] = gt_boxes[:, 3]
            gt_boxes[:, 3] = width
            corners_lidar_gt = LB(infos['gt_boxes'], origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
            corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt], axis=0)
            gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool_)
            pred_flag = np.concatenate([pred_flag, np.logical_not(gt_flag)], axis=0)
            scores = scores + [0 for _ in range(infos['gt_boxes'].shape[0])]
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        # 读取并缩放图像
        imgs = []
        for view in views:
            img_path = infos['cams'][view]['data_path']
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"图像文件不存在：{img_path}")
            h, w = img.shape[:2]
            scale = target_img_height / h
            target_width = int(w * scale)
            img_resized = cv2.resize(img, (target_width, target_img_height))
            imgs.append(img_resized)

        # 绘制BEV
        canvas = np.zeros((canva_size, canva_size, 3), dtype=np.uint8)
        lidar_points = np.fromfile(infos['lidar_path'], dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 6)[:, :3]
        lidar_points[:, 1] = -lidar_points[:, 1]
        lidar_points[:, :2] = (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
        for p in lidar_points:
            if check_point_in_img(p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                color = depth2color(p[2])
                cv2.circle(canvas, (int(p[0]), int(p[1])), radius=0, color=color, thickness=1)

        # 绘制实例框（略）
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = (center_bev + show_range) / show_range / 2.0 * canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
        head_canvas = head_canvas.astype(np.int32)
        for rid in sort_ids:
            score = scores[rid]
            if score < args.vis_thred and pred_flag[rid]:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = color_map[int(pred_flag[rid])]
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    bottom_corners_bev[rid, index[0]],
                    bottom_corners_bev[rid, index[1]],
                    [int(color[0] * score), int(color[1] * score), int(color[2] * score)],
                    thickness=1)
            cv2.line(
                canvas,
                center_canvas[rid],
                head_canvas[rid],
                [int(color[0] * score), int(color[1] * score), int(color[2] * score)],
                1,
                lineType=8)

        # 融合图像和BEV（核心修改）
        top_imgs = imgs[:3]
        top_height = target_img_height
        top_width = sum(img.shape[1] for img in top_imgs)
        bottom_imgs = [cv2.flip(img, 1) for img in imgs[3:]]
        bottom_height = target_img_height
        bottom_width = sum(img.shape[1] for img in bottom_imgs)
        bev_height = canva_size * scale_factor

        # 关键修改1：强制总宽度不小于BEV所需宽度（canva_size * scale_factor）
        min_total_width = canva_size * scale_factor  # BEV区域缩放前的最小宽度
        total_width = max(top_width, bottom_width, min_total_width)
        total_height = top_height + bottom_height + bev_height
        fused_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # 放置顶部图像
        current_x = 0
        for img in top_imgs:
            fused_img[:top_height, current_x:current_x + img.shape[1], :] = img
            current_x += img.shape[1]
        # 放置底部图像
        current_x = 0
        bev_end_y = top_height + bev_height
        for img in bottom_imgs:
            fused_img[bev_end_y:bev_end_y + bottom_height, current_x:current_x + img.shape[1], :] = img
            current_x += img.shape[1]

        # 缩放总画布
        scaled_width = int(total_width / scale_factor)
        scaled_height = int(total_height / scale_factor)
        fused_img_scaled = cv2.resize(fused_img, (scaled_width, scaled_height))

        # 关键修改2：确保BEV放置区域宽度足够
        bev_start_x = max(0, (scaled_width - canva_size) // 2)  # 避免负坐标
        # 强制BEV区域宽度为canva_size（防止超出画布）
        bev_end_x = min(bev_start_x + canva_size, scaled_width)
        bev_start_x = bev_end_x - canva_size  # 确保宽度刚好为canva_size

        bev_start_y = int(top_height / scale_factor)
        # 放置BEV画布（确保尺寸匹配）
        fused_img_scaled[
            bev_start_y:bev_start_y + canva_size,
            bev_start_x:bev_start_x + canva_size, :
        ] = canvas

        # 初始化视频写入器
        if args.format == 'video' and vout is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vout = cv2.VideoWriter(
                os.path.join(vis_dir, f'{args.video_prefix}.mp4'),
                fourcc,
                args.fps,
                (scaled_width, scaled_height)
            )

        # 保存结果
        if args.format == 'image':
            cv2.imwrite(os.path.join(vis_dir, f'{infos["token"]}.jpg'), fused_img_scaled)
        elif args.format == 'video':
            vout.write(fused_img_scaled)

    if args.format == 'video' and vout is not None:
        vout.release()
    print(f"可视化完成，结果保存至：{vis_dir}")


if __name__ == '__main__':
    main()