#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从.npz文件生成人脸数据库
"""

import numpy as np
import argparse
import os

def load_npz_database(npz_path):
    """
    从.npz文件加载人脸数据库
    期望格式: names (字符串数组), embeddings (特征向量数组, shape: [num_people, 512])
    """
    print(f"加载.npz文件: {npz_path}")

    data = np.load(npz_path)

    if 'names' not in data or 'embeddings' not in data:
        print("错误: .npz文件必须包含 'names' 和 'embeddings' 字段")
        return None, None

    names = data['names']
    embeddings = data['embeddings']

    print(f"加载了 {len(names)} 个人脸数据")
    print(f"特征向量形状: {embeddings.shape}")

    # 确保特征向量是float32类型并进行L2归一化
    embeddings = embeddings.astype(np.float32)

    # 对每个特征向量进行L2归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    embeddings = embeddings / norms

    return names, embeddings

def save_database(names, embeddings, output_path):
    """
    保存数据库到二进制文件
    格式: num_people + (name_len + name + feature[512]) * num_people
    """
    print(f"\n保存数据库到: {output_path}")

    with open(output_path, 'wb') as f:
        # 写入人数
        num_people = len(names)
        f.write(np.int32(num_people).tobytes())

        print(f"保存 {num_people} 个人脸数据")

        for i in range(num_people):
            name = str(names[i])  # 确保是字符串
            feature = embeddings[i]

            # 写入人名长度和人名
            name_bytes = name.encode('utf-8')
            name_len = len(name_bytes)
            f.write(np.int32(name_len).tobytes())
            f.write(name_bytes)

            # 写入特征向量 (512个float)
            f.write(feature.astype(np.float32).tobytes())

            print(f"  {i+1}/{num_people}: {name}")

    print(f"数据库保存完成，大小: {os.path.getsize(output_path)} 字节")

def save_info_file(names, output_path):
    """
    保存信息文件
    """
    info_path = output_path.replace('.bin', '_info.txt')
    print(f"保存信息文件到: {info_path}")

    with open(info_path, 'w', encoding='utf-8') as f:
        for i, name in enumerate(names):
            f.write(f"{i}: {name}\n")

    print("信息文件保存完成")

def main():
    parser = argparse.ArgumentParser(description='从.npz文件生成人脸数据库')
    parser.add_argument('--npz_file', required=True, help='.npz文件路径')
    parser.add_argument('--output', default='face_database.bin', help='输出数据库文件路径')

    args = parser.parse_args()

    if not os.path.exists(args.npz_file):
        print(f"错误: .npz文件不存在 - {args.npz_file}")
        return

    print("=== 从.npz文件生成人脸数据库 ===")
    print(f"输入文件: {args.npz_file}")
    print(f"输出文件: {args.output}")
    print()

    # 1. 加载.npz文件
    names, embeddings = load_npz_database(args.npz_file)

    if names is None or embeddings is None:
        return

    # 2. 保存数据库
    save_database(names, embeddings, args.output)
    save_info_file(names, args.output)

    print("\n=== 处理完成 ===")
    print(f"数据库文件: {args.output}")
    print(f"信息文件: {args.output.replace('.bin', '_info.txt')}")

if __name__ == '__main__':
    main()
