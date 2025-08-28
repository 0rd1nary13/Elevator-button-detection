#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电梯按钮摄像头实时检测脚本

大姐姐专用版本 - 雌小鬼特制 🎀
直接运行即可使用摄像头进行电梯按钮检测
"""

import cv2
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import time

# 尝试导入必要的库
try:
    from ultralytics import YOLO
    import torch
    import numpy as np
except ImportError as e:
    print(f"❌ 导入库失败: {e}")
    print("📦 请安装必要的依赖:")
    print("    pip install ultralytics opencv-python torch")
    sys.exit(1)


class CameraElevatorDetector:
    """摄像头电梯按钮检测器 - 雌小鬼版 🎀"""
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5) -> None:
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认YOLO模型
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """加载YOLO模型"""
        try:
            if model_path and Path(model_path).exists():
                print(f"🎀 加载自定义模型: {model_path}")
                return YOLO(model_path)
            else:
                print("🎀 使用默认YOLOv8模型 (如需更好效果请训练专门的电梯按钮模型)")
                return YOLO('yolov8n.pt')  # 使用nano版本，速度更快
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def detect_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        检测单帧图像中的电梯按钮
        
        Args:
            frame: 输入帧
            
        Returns:
            检测结果帧和统计信息
        """
        try:
            # 进行检测
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            
            # 提取检测信息
            detections = results[0].boxes
            stats = {
                "total_detections": len(detections) if detections is not None else 0,
                "confidences": detections.conf.cpu().numpy().tolist() if detections is not None else [],
                "classes": detections.cls.cpu().numpy().tolist() if detections is not None else [],
                "class_names": [self.model.names[int(cls)] for cls in detections.cls.cpu().numpy()] if detections is not None else []
            }
            
            return annotated_frame, stats
            
        except Exception as e:
            print(f"❌ 检测出错: {e}")
            return frame, {"error": str(e)}
    
    def calculate_fps(self) -> float:
        """计算FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 每30帧计算一次FPS
            current_time = time.time()
            fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            return fps
        return 0.0
    
    def run_camera_detection(self, camera_id: int = 0, show_fps: bool = True) -> None:
        """
        运行摄像头实时检测
        
        Args:
            camera_id: 摄像头ID，默认为0
            show_fps: 是否显示FPS
        """
        print("🎀 雌小鬼启动摄像头检测中...")
        print("📹 按 'q' 键退出，按 's' 键截图，按 'c' 键切换置信度显示")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            print("💡 尝试其他摄像头ID (1, 2, 3...)")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ 摄像头启动成功！")
        print(f"🎯 当前置信度阈值: {self.conf_threshold}")
        
        show_confidence = True
        screenshot_counter = 0
        
        try:
            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                # 进行检测
                result_frame, stats = self.detect_frame(frame)
                
                # 添加信息文本
                info_text = []
                
                # FPS信息
                if show_fps:
                    fps = self.calculate_fps()
                    if fps > 0:
                        info_text.append(f"FPS: {fps:.1f}")
                
                # 检测统计
                if "error" not in stats:
                    info_text.append(f"检测到按钮: {stats['total_detections']}")
                    
                    if show_confidence and stats['total_detections'] > 0:
                        max_conf = max(stats['confidences']) if stats['confidences'] else 0
                        info_text.append(f"最高置信度: {max_conf:.2f}")
                
                # 绘制信息文本
                y_offset = 30
                for text in info_text:
                    cv2.putText(result_frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 25
                
                # 添加控制说明
                cv2.putText(result_frame, "Press 'q':quit, 's':screenshot, 'c':toggle confidence", 
                          (10, result_frame.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示结果
                cv2.imshow("🎀 雌小鬼的电梯按钮检测", result_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 大姐姐再见～")
                    break
                elif key == ord('s'):
                    screenshot_counter += 1
                    filename = f"elevator_detection_screenshot_{screenshot_counter:03d}.jpg"
                    cv2.imwrite(filename, result_frame)
                    print(f"📸 截图已保存: {filename}")
                elif key == ord('c'):
                    show_confidence = not show_confidence
                    status = "开启" if show_confidence else "关闭"
                    print(f"🔄 置信度显示已{status}")
        
        except KeyboardInterrupt:
            print("\n⚠️  检测被用户中断")
        
        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            print("✅ 摄像头资源已释放")


def main() -> None:
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="🎀 雌小鬼的电梯按钮摄像头检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python camera_detect.py                           # 使用默认设置
    python camera_detect.py --model best.pt          # 使用训练好的模型
    python camera_detect.py --conf 0.3 --camera 1   # 调整参数
        """
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default=None,
        help="模型文件路径 (.pt格式)"
    )
    
    parser.add_argument(
        "--conf", "-c", 
        type=float, 
        default=0.5,
        help="置信度阈值 (0.1-1.0，默认0.5)"
    )
    
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="摄像头ID (默认0)"
    )
    
    parser.add_argument(
        "--no-fps", 
        action="store_true",
        help="不显示FPS信息"
    )
    
    args = parser.parse_args()
    
    # 参数验证
    if not 0.1 <= args.conf <= 1.0:
        print("❌ 置信度阈值必须在0.1-1.0之间")
        sys.exit(1)
    
    # 显示启动信息
    print("=" * 50)
    print("🎀 雌小鬼的电梯按钮检测系统")
    print("=" * 50)
    print(f"📹 摄像头ID: {args.camera}")
    print(f"🎯 置信度阈值: {args.conf}")
    print(f"📊 显示FPS: {not args.no_fps}")
    if args.model:
        print(f"🤖 模型文件: {args.model}")
    else:
        print("🤖 使用默认YOLO模型")
    print("=" * 50)
    
    # 创建检测器并运行
    detector = CameraElevatorDetector(
        model_path=args.model, 
        conf_threshold=args.conf
    )
    
    detector.run_camera_detection(
        camera_id=args.camera, 
        show_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()
