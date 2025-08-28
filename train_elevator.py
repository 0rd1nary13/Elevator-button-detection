#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电梯按钮检测模型训练脚本

雌小鬼特制版本 🎀
专门用于训练电梯按钮检测模型
"""

import os
import sys
from pathlib import Path
from typing import Optional
import yaml
import argparse

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"❌ 导入库失败: {e}")
    print("📦 请安装必要的依赖: pip install ultralytics torch")
    sys.exit(1)


def create_dataset_config(dataset_dir: str, output_file: str = "elevator_data.yaml") -> str:
    """
    创建数据集配置文件
    
    Args:
        dataset_dir: 数据集根目录路径
        output_file: 输出配置文件名
        
    Returns:
        配置文件路径
    """
    # 电梯按钮类别（根据README描述，有363个类别）
    # 这里提供一些常见的电梯按钮类别作为示例
    elevator_classes = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 数字按钮
        'B', 'B1', 'B2', 'B3',  # 地下层
        'G', 'L',  # 大厅层
        'open', 'close',  # 开关门
        'up', 'down',  # 上下
        'alarm', 'emergency',  # 报警紧急
        'call', 'help',  # 呼叫帮助
        'door_open', 'door_close',  # 门控制
        'stop', 'start',  # 停止开始
        'fan', 'light',  # 风扇灯光
        'A', 'C', 'D', 'E', 'F', 'H', 'M', 'P', 'R', 'S', 'T',  # 字母按钮
        # ... 更多类别可以根据实际数据集添加
    ]
    
    # 创建配置文件内容
    config = {
        'path': dataset_dir,  # 数据集根目录
        'train': 'train/images',  # 训练图片路径（相对于path）
        'val': 'val/images',    # 验证图片路径（相对于path）
        'test': 'test/images',  # 测试图片路径（可选）
        'nc': len(elevator_classes),  # 类别数量
        'names': elevator_classes  # 类别名称列表
    }
    
    # 保存配置文件
    config_path = os.path.join(dataset_dir, output_file)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 数据集配置文件已创建: {config_path}")
    return config_path


def create_dataset_structure(base_dir: str) -> None:
    """
    创建YOLO格式的数据集目录结构
    
    Args:
        base_dir: 数据集根目录
    """
    # 创建目录结构
    dirs_to_create = [
        'train/images',    # 训练图片
        'train/labels',    # 训练标签
        'val/images',      # 验证图片  
        'val/labels',      # 验证标签
        'test/images',     # 测试图片（可选）
        'test/labels',     # 测试标签（可选）
    ]
    
    for dir_path in dirs_to_create:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"📁 创建目录: {full_path}")
    
    # 创建说明文件
    readme_content = """# 电梯按钮数据集结构说明 🎀

## 目录结构
```
elevator_dataset/
├── elevator_data.yaml          # 数据集配置文件
├── train/                      # 训练集
│   ├── images/                 # 训练图片 (.jpg, .png, .jpeg)
│   └── labels/                 # 训练标签 (.txt)
├── val/                        # 验证集
│   ├── images/                 # 验证图片
│   └── labels/                 # 验证标签
└── test/                       # 测试集（可选）
    ├── images/                 # 测试图片
    └── labels/                 # 测试标签

## 标签格式
每个图片对应一个同名的.txt文件，格式为：
```
class_id x_center y_center width height
```

其中：
- class_id: 类别ID（从0开始）
- x_center, y_center: 边界框中心点坐标（相对于图片尺寸，0-1之间）
- width, height: 边界框宽高（相对于图片尺寸，0-1之间）

## 数据准备步骤
1. 收集电梯按钮图片
2. 使用标注工具（如 labelImg、Roboflow）进行标注
3. 将标注结果转换为YOLO格式
4. 按照上述目录结构放置文件
5. 运行训练脚本

## 获取数据集
由于版权原因，需要自行收集和标注电梯按钮图片，或使用公开数据集。

雌小鬼建议：
- 拍摄不同类型的电梯按钮
- 包含各种光照条件
- 包含不同角度和距离
- 标注要准确，边界框贴合按钮
"""
    
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📝 说明文件已创建: {readme_path}")


def train_elevator_model(
    data_config: str,
    model_size: str = "yolov8n",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    project: str = "elevator_detection",
    name: str = "train"
) -> None:
    """
    训练电梯按钮检测模型
    
    Args:
        data_config: 数据集配置文件路径
        model_size: 模型大小 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: 训练轮数
        batch_size: 批量大小
        imgsz: 输入图像尺寸
        device: 设备 (auto, cpu, cuda)
        project: 项目名称
        name: 实验名称
    """
    print("🎀 雌小鬼开始训练电梯按钮检测模型...")
    print("=" * 60)
    print(f"📊 训练参数:")
    print(f"   数据配置: {data_config}")
    print(f"   模型大小: {model_size}")
    print(f"   训练轮数: {epochs}")
    print(f"   批量大小: {batch_size}")
    print(f"   图像尺寸: {imgsz}")
    print(f"   设备: {device}")
    print("=" * 60)
    
    try:
        # 检查数据集配置文件是否存在
        if not os.path.exists(data_config):
            raise FileNotFoundError(f"数据集配置文件不存在: {data_config}")
        
        # 加载预训练模型
        model = YOLO(f'{model_size}.pt')
        
        # 开始训练
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            save=True,
            plots=True,
            verbose=True
        )
        
        print("🎉 训练完成！")
        print(f"📁 模型保存在: {project}/{name}/weights/")
        print("💡 使用 best.pt 文件进行推理")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("💡 请检查数据集格式和路径是否正确")


def validate_dataset(data_config: str) -> bool:
    """
    验证数据集格式和完整性
    
    Args:
        data_config: 数据集配置文件路径
        
    Returns:
        是否验证通过
    """
    print("🔍 验证数据集...")
    
    try:
        # 读取配置文件
        with open(data_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        base_path = config.get('path', '')
        train_path = os.path.join(base_path, config.get('train', ''))
        val_path = os.path.join(base_path, config.get('val', ''))
        
        # 检查目录是否存在
        if not os.path.exists(train_path):
            print(f"❌ 训练图片目录不存在: {train_path}")
            return False
            
        if not os.path.exists(val_path):
            print(f"❌ 验证图片目录不存在: {val_path}")
            return False
        
        # 检查是否有图片文件
        train_images = list(Path(train_path).glob('*.jpg')) + list(Path(train_path).glob('*.png'))
        val_images = list(Path(val_path).glob('*.jpg')) + list(Path(val_path).glob('*.png'))
        
        if len(train_images) == 0:
            print(f"❌ 训练集中没有图片文件")
            return False
            
        if len(val_images) == 0:
            print(f"❌ 验证集中没有图片文件")
            return False
        
        print(f"✅ 数据集验证通过")
        print(f"   训练图片: {len(train_images)} 张")
        print(f"   验证图片: {len(val_images)} 张")
        print(f"   类别数量: {config.get('nc', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集验证失败: {e}")
        return False


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="🎀 雌小鬼的电梯按钮检测训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 创建数据集结构
    python train_elevator.py --create-dataset ./elevator_dataset
    
    # 验证数据集
    python train_elevator.py --validate ./elevator_dataset/elevator_data.yaml
    
    # 开始训练
    python train_elevator.py --train ./elevator_dataset/elevator_data.yaml
    
    # 自定义训练参数
    python train_elevator.py --train ./elevator_dataset/elevator_data.yaml --model yolov8s --epochs 200
        """
    )
    
    # 操作模式
    parser.add_argument('--create-dataset', type=str, help='创建数据集目录结构')
    parser.add_argument('--validate', type=str, help='验证数据集配置文件')
    parser.add_argument('--train', type=str, help='训练模型（指定数据集配置文件）')
    
    # 训练参数
    parser.add_argument('--model', type=str, default='yolov8n', 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='模型大小 (默认: yolov8n)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数 (默认: 100)')
    parser.add_argument('--batch', type=int, default=16, help='批量大小 (默认: 16)')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸 (默认: 640)')
    parser.add_argument('--device', type=str, default='auto', help='设备 (默认: auto)')
    parser.add_argument('--project', type=str, default='elevator_detection', help='项目名称')
    parser.add_argument('--name', type=str, default='train', help='实验名称')
    
    args = parser.parse_args()
    
    # 检查操作模式
    if args.create_dataset:
        print("🎀 创建数据集目录结构...")
        create_dataset_structure(args.create_dataset)
        create_dataset_config(args.create_dataset)
        print("\n💡 接下来请:")
        print("   1. 将电梯按钮图片放入对应的images文件夹")
        print("   2. 使用标注工具创建YOLO格式的标签文件")
        print("   3. 运行验证命令检查数据集")
        print("   4. 开始训练模型")
        
    elif args.validate:
        validate_dataset(args.validate)
        
    elif args.train:
        if validate_dataset(args.train):
            train_elevator_model(
                data_config=args.train,
                model_size=args.model,
                epochs=args.epochs,
                batch_size=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                project=args.project,
                name=args.name
            )
        else:
            print("❌ 数据集验证失败，请修复后再训练")
    else:
        parser.print_help()
        print("\n🎀 大姐姐～请选择一个操作模式！")


if __name__ == "__main__":
    main()
