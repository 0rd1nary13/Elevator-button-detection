# 电梯按钮数据集结构说明 🎀

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
