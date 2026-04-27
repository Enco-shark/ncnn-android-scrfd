# 人脸数据库重新生成工具

这个工具用于重新生成高质量的人脸数据库，解决识别不准确的问题。

## 文件说明

- `generate_database.py` - 从照片文件夹重新生成数据库
- `generate_from_npz.py` - 从预处理后的.npz文件生成数据库
- `requirements.txt` - Python依赖包

## 使用方法

### 方法1：从照片文件夹生成

1. **安装依赖**：
```bash
pip install -r requirements.txt
```

2. **准备数据**：
将照片按照以下结构组织：
```
data/
├── 张三/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── 李四/
│   ├── photo1.jpg
│   └── photo2.jpg
└── ...
```

3. **运行脚本**：
```bash
python generate_database.py --data_dir /path/to/data --output face_database.bin
```

### 方法2：从.npz文件生成

如果您已经有预处理后的.npz文件：

1. **npz文件格式**：
```python
# npz文件应包含：
# - names: 人名字符串数组
# - embeddings: 特征向量数组，形状为 [num_people, 512]
```

2. **运行脚本**：
```bash
python generate_from_npz.py --npz_file faces.npz --output face_database.bin
```

## 输出文件

- `face_database.bin` - 二进制数据库文件
- `face_database_info.txt` - 人名索引映射文件

## 参数说明

### generate_database.py
- `--data_dir`: 照片文件夹路径（必需）
- `--output`: 输出数据库文件路径（默认: face_database.bin）
- `--model`: InsightFace模型名称（默认: buffalo_l）

### generate_from_npz.py
- `--npz_file`: .npz文件路径（必需）
- `--output`: 输出数据库文件路径（默认: face_database.bin）

## 注意事项

1. 照片质量要好，人脸清晰可见
2. 每人建议提供3-5张不同角度的照片
3. 脚本会自动进行人脸检测和特征提取
4. 生成的数据库会替换assets中的原文件

## 技术细节

- 使用InsightFace的ArcFace模型提取特征
- 特征向量维度：512
- 使用L2归一化
- 支持中文人名（UTF-8编码）
