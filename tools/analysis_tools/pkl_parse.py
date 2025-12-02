import pickle
import os
import numpy as np

def print_dict(d, indent=0):
    """递归打印嵌套字典，增强可读性"""
    for key, value in d.items():
        print(' ' * indent + f"{key}: ", end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 4)
        elif isinstance(value, list):
            if len(value) == 0:
                print("[]")
            elif isinstance(value[0], dict) and len(value) > 3:
                # 列表包含多个字典时，只打印前3个示例
                print(f"[共 {len(value)} 个元素，前3个示例如下]")
                for i, item in enumerate(value[:3]):
                    print(' ' * (indent + 4) + f"第{i+1}个元素:")
                    print_dict(item, indent + 8)
            else:
                # 普通列表直接打印
                print(value)
        elif isinstance(value, np.ndarray):
            # 数组打印形状和部分内容
            print(f"ndarray(shape={value.shape}, dtype={value.dtype})，前5个元素: {value[:5]}")
        else:
            # 基本类型直接打印
            print(value)

def parse_nuscenes_pkl(pkl_path):
    """解析nuscenes格式的pkl文件，打印所有关键信息"""
    # 1. 加载pkl文件
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 2. 解析metadata（数据集基本信息）
    print("="*50)
    print("1. 数据集元信息 (metadata):")
    print("="*50)
    if 'metadata' in dataset:
        print_dict(dataset['metadata'])
    else:
        print("未找到metadata字段")
    
    # 3. 解析infos（样本详细信息，以第一个样本为例）
    print("\n" + "="*50)
    print("2. 样本详细信息 (infos) - 第一个样本:")
    print("="*50)
    if 'infos' in dataset and len(dataset['infos']) > 0:
        info = dataset['infos'][0]  # 取第一个样本
        print_dict(info, indent=4)
    else:
        print("未找到infos字段或infos为空")
    
    print("\n" + "="*50)
    print("2. 样本详细信息 (infos) - 第2个样本:")
    print("="*50)
    if 'infos' in dataset and len(dataset['infos']) > 0:
        info = dataset['infos'][1]  # 取第一个样本
        print_dict(info, indent=4)
    else:
        print("未找到infos字段或infos为空")

if __name__ == "__main__":
    # 配置路径（替换为你的pkl文件路径）
    data_root = '/workspace/BEV/BEVDet/data/nuscenes'
    pkl_file = os.path.join(data_root, 'bevdetv3-nuscenes_infos_val.pkl')
    
    # 解析并打印所有数据
    parse_nuscenes_pkl(pkl_file)