#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸数据库重新生成脚本
使用高质量的特征提取方法重新生成face_database.bin
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import warnings
warnings.filterwarnings('ignore')

def load_faces_from_directory(data_dir):
    """
    从目录结构加载人脸数据
    data_dir/
    ├── person1/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── photo3.jpg
    ├── person2/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    └── ...
    """
    faces_data = []

    print(f"扫描目录: {data_dir}")

    for person_dir in sorted(os.listdir(data_dir)):
        person_path = os.path.join(data_dir, person_dir)
        if not os.path.isdir(person_path):
            continue

        person_name = person_dir
        person_images = []

        print(f"处理人员: {person_name}")

        for image_file in sorted(os.listdir(person_path)):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(person_path, image_file)
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        person_images.append(img)
                        print(f"  加载图片: {image_file}")
                    else:
                        print(f"  跳过无效图片: {image_file}")
                except Exception as e:
                    print(f"  加载图片失败 {image_file}: {e}")

        if person_images:
            faces_data.append({
                'name': person_name,
                'images': person_images
            })
            print(f"  共加载 {len(person_images)} 张图片")
        else:
            print(f"  警告: {person_name} 没有有效图片")

    return faces_data

def extract_face_features(faces_data, model_name='buffalo_l'):
    """
    使用InsightFace提取高质量的人脸特征
    """
    print("\n初始化人脸分析模型...")

    # 使用CPU模式
    app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("开始提取人脸特征...")

    processed_data = []

    for person_data in faces_data:
        person_name = person_data['name']
        images = person_data['images']

        print(f"\n处理人员: {person_name} ({len(images)} 张图片)")

        person_features = []

        for i, img in enumerate(images):
            try:
                # 检测人脸
                faces = app.get(img)

                if len(faces) == 0:
                    print(f"  图片 {i+1}: 未检测到人脸")
                    continue
                elif len(faces) > 1:
                    print(f"  图片 {i+1}: 检测到 {len(faces)} 个人脸，使用最大的人脸")
                    # 选择最大的人脸
                    faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)

                # 提取特征
                face = faces[0]
                feature = face.embedding.astype(np.float32)

                # L2归一化
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm

                person_features.append(feature)
                print(f"  图片 {i+1}: 特征提取成功")

            except Exception as e:
                print(f"  图片 {i+1}: 特征提取失败 - {e}")
                continue

        if person_features:
            # 计算平均特征向量
            avg_feature = np.mean(person_features, axis=0)
            # 再次归一化
            norm = np.linalg.norm(avg_feature)
            if norm > 0:
                avg_feature = avg_feature / norm

            processed_data.append({
                'name': person_name,
                'feature': avg_feature,
                'num_images': len(person_features)
            })

            print(f"  成功提取 {len(person_features)} 个特征，平均特征已计算")
        else:
            print(f"  警告: {person_name} 未能提取到任何特征")

    return processed_data

def save_database(processed_data, output_path):
    """
    保存数据库到二进制文件
    格式: num_people + (name_len + name + feature[512]) * num_people
    """
    print(f"\n保存数据库到: {output_path}")

    with open(output_path, 'wb') as f:
        # 写入人数
        num_people = len(processed_data)
        f.write(np.int32(num_people).tobytes())

        print(f"保存 {num_people} 个人脸数据")

        for i, person_data in enumerate(processed_data):
            name = person_data['name']
            feature = person_data['feature']
            num_images = person_data['num_images']

            # 写入人名长度和人名
            name_bytes = name.encode('utf-8')
            name_len = len(name_bytes)
            f.write(np.int32(name_len).tobytes())
            f.write(name_bytes)

            # 写入特征向量 (512个float)
            f.write(feature.astype(np.float32).tobytes())

            print(f"  {i+1}/{num_people}: {name} ({num_images} 张图片)")

    print(f"数据库保存完成，大小: {os.path.getsize(output_path)} 字节")

def save_info_file(processed_data, output_path):
    """
    保存信息文件
    """
    info_path = output_path.replace('.bin', '_info.txt')
    print(f"保存信息文件到: {info_path}")

    with open(info_path, 'w', encoding='utf-8') as f:
        for i, person_data in enumerate(processed_data):
            f.write(f"{i}: {person_data['name']}\n")

    print("信息文件保存完成")

def main():
    parser = argparse.ArgumentParser(description='重新生成人脸数据库')
    parser.add_argument('--data_dir', required=True, help='包含每个人照片的文件夹路径')
    parser.add_argument('--output', default='face_database.bin', help='输出数据库文件路径')
    parser.add_argument('--model', default='buffalo_l', help='InsightFace模型名称')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在 - {args.data_dir}")
        return

    print("=== 人脸数据库重新生成工具 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"输出文件: {args.output}")
    print(f"使用模型: {args.model}")
    print()

    # 1. 加载人脸数据
    faces_data = load_faces_from_directory(args.data_dir)

    if not faces_data:
        print("错误: 未找到任何有效的人脸数据")
        return

    print(f"\n找到 {len(faces_data)} 个人物数据")

    # 2. 提取特征
    processed_data = extract_face_features(faces_data, args.model)

    if not processed_data:
        print("错误: 未能提取到任何特征")
        return

    print(f"\n成功处理 {len(processed_data)} 个人物数据")

    # 3. 保存数据库
    save_database(processed_data, args.output)
    save_info_file(processed_data, args.output)

    print("\n=== 处理完成 ===")
    print(f"数据库文件: {args.output}")
    print(f"信息文件: {args.output.replace('.bin', '_info.txt')}")

if __name__ == '__main__':
    main()
