#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电梯按钮检测系统 Web 界面

使用雌小鬼风格的高质量代码实现 🎀
"""

from typing import Optional, Tuple, Dict, Any
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
import os
from pathlib import Path

# 尝试导入 ultralytics，如果没有就提示安装
try:
    from ultralytics import YOLO
except ImportError:
    st.error("请先安装 ultralytics: pip install ultralytics")
    st.stop()


class ElevatorButtonDetector:
    """电梯按钮检测器类 - 雌小鬼特制版本 🎀"""
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        初始化检测器
        
        Args:
            model_path: 模型权重文件路径
        """
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = model_path
        self._load_model()
    
    def _load_model(self) -> None:
        """加载YOLO模型"""
        try:
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                st.success(f"模型加载成功: {self.model_path}")
            else:
                # 使用预训练模型作为fallback
                self.model = YOLO('yolov8n.pt')
                st.warning("使用默认YOLOv8模型，请上传训练好的电梯按钮检测模型以获得更好效果")
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            self.model = None
    
    def detect_image(self, image: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        对图像进行按钮检测
        
        Args:
            image: 输入图像 (numpy array)
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果图像和统计信息
        """
        if self.model is None:
            return image, {"error": "模型未加载"}
        
        try:
            # 进行检测
            results = self.model(image, conf=conf_threshold, verbose=False)
            
            # 绘制检测结果
            annotated_image = results[0].plot()
            
            # 统计检测信息
            detections = results[0].boxes
            stats = {
                "total_detections": len(detections) if detections is not None else 0,
                "confidences": detections.conf.cpu().numpy().tolist() if detections is not None else [],
                "classes": detections.cls.cpu().numpy().tolist() if detections is not None else [],
                "class_names": [self.model.names[int(cls)] for cls in detections.cls.cpu().numpy()] if detections is not None else []
            }
            
            return annotated_image, stats
            
        except Exception as e:
            st.error(f"检测过程出错: {str(e)}")
            return image, {"error": str(e)}


def create_sidebar() -> Dict[str, Any]:
    """创建侧边栏配置"""
    st.sidebar.title("🎀 雌小鬼的电梯按钮检测系统")
    st.sidebar.markdown("---")
    
    # 模型配置
    st.sidebar.subheader("模型设置")
    uploaded_model = st.sidebar.file_uploader(
        "上传训练好的模型文件 (.pt)", 
        type=['pt'],
        help="上传你训练好的电梯按钮检测模型"
    )
    
    conf_threshold = st.sidebar.slider(
        "置信度阈值", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # 检测模式选择
    st.sidebar.subheader("检测模式")
    detection_mode = st.sidebar.selectbox(
        "选择检测方式",
        ["图片上传", "摄像头实时检测", "视频文件"]
    )
    
    return {
        "uploaded_model": uploaded_model,
        "conf_threshold": conf_threshold,
        "detection_mode": detection_mode
    }


def handle_image_upload(detector: ElevatorButtonDetector, conf_threshold: float) -> None:
    """处理图片上传模式"""
    st.header("📷 图片检测模式")
    
    uploaded_file = st.file_uploader(
        "选择要检测的图片", 
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        try:
            # 读取图片
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # 显示原图
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原始图片")
                st.image(image, use_container_width=True)
            
            # 进行检测
            with st.spinner("正在检测电梯按钮..."):
                result_image, stats = detector.detect_image(image_np, conf_threshold)
            
            with col2:
                st.subheader("检测结果")
                st.image(result_image, use_container_width=True)
            
            # 显示统计信息
            if "error" not in stats:
                st.success(f"检测完成！发现 {stats['total_detections']} 个按钮")
                
                if stats['total_detections'] > 0:
                    st.subheader("检测详情")
                    for i, (conf, class_name) in enumerate(zip(stats['confidences'], stats['class_names'])):
                        st.write(f"按钮 {i+1}: {class_name} (置信度: {conf:.2f})")
            else:
                st.error(f"检测失败: {stats['error']}")
                
        except Exception as e:
            st.error(f"图片处理错误: {str(e)}")


def handle_camera_detection(detector: ElevatorButtonDetector, conf_threshold: float) -> None:
    """处理摄像头实时检测模式"""
    st.header("📹 摄像头实时检测")
    st.info("大姐姐～点击下面的按钮开始实时检测！")
    
    # 创建控制按钮
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_camera = st.button("📸 启动摄像头", type="primary")
    
    with col2:
        stop_camera = st.button("⏹️ 停止检测")
    
    with col3:
        capture_frame = st.button("📷 截取当前画面")
    
    # 摄像头控制逻辑
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if start_camera:
        st.session_state.camera_active = True
    
    if stop_camera:
        st.session_state.camera_active = False
    
    # 实时检测逻辑
    if st.session_state.camera_active:
        try:
            # 创建显示区域
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            # 打开摄像头
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("无法打开摄像头！请检查摄像头连接。")
                st.session_state.camera_active = False
                return
            
            # 设置摄像头分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # 实时检测循环
            frame_count = 0
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("无法读取摄像头画面")
                    break
                
                # 每5帧检测一次，提高性能
                if frame_count % 5 == 0:
                    # BGR转RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 进行检测
                    result_frame, stats = detector.detect_image(frame_rgb, conf_threshold)
                    
                    # 显示结果
                    frame_placeholder.image(result_frame, channels="RGB", use_container_width=True)
                    
                    # 显示统计信息
                    if "error" not in stats:
                        stats_info = f"🎯 检测到 {stats['total_detections']} 个按钮"
                        if stats['total_detections'] > 0:
                            stats_info += f" | 类别: {', '.join(set(stats['class_names']))}"
                        stats_placeholder.info(stats_info)
                    
                    # 截取画面功能
                    if capture_frame and 'captured_frame' not in st.session_state:
                        st.session_state.captured_frame = result_frame
                        st.success("画面已截取！")
                
                frame_count += 1
                
                # 添加小延迟避免过度消耗资源
                import time
                time.sleep(0.03)
            
            # 释放摄像头
            cap.release()
            
        except Exception as e:
            st.error(f"摄像头检测出错: {str(e)}")
            st.session_state.camera_active = False
    
    # 显示截取的画面
    if 'captured_frame' in st.session_state:
        st.subheader("📸 截取的画面")
        st.image(st.session_state.captured_frame, use_container_width=True)
        if st.button("清除截图"):
            del st.session_state.captured_frame


def handle_video_upload(detector: ElevatorButtonDetector, conf_threshold: float) -> None:
    """处理视频文件检测模式"""
    st.header("🎬 视频检测模式")
    
    uploaded_video = st.file_uploader(
        "上传视频文件", 
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_video is not None:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        try:
            # 显示原始视频
            st.subheader("原始视频")
            st.video(uploaded_video)
            
            # 处理视频
            process_video = st.button("🚀 开始检测", type="primary")
            
            if process_video:
                with st.spinner("正在处理视频，请耐心等待..."):
                    # 这里可以添加视频处理逻辑
                    st.info("视频检测功能开发中... 敬请期待！")
        
        finally:
            # 清理临时文件
            if os.path.exists(video_path):
                os.unlink(video_path)


def main() -> None:
    """主程序入口"""
    # 页面配置
    st.set_page_config(
        page_title="电梯按钮检测系统",
        page_icon="🎀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 主标题
    st.title("🎀 雌小鬼的电梯按钮检测系统")
    st.markdown("---")
    st.markdown("### 大姐姐～欢迎使用我开发的电梯按钮检测系统呢！")
    
    # 创建侧边栏
    config = create_sidebar()
    
    # 处理模型上传
    model_path = None
    if config["uploaded_model"] is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(config["uploaded_model"].read())
            model_path = tmp_file.name
    
    # 初始化检测器
    detector = ElevatorButtonDetector(model_path)
    
    # 根据模式选择不同的处理函数
    if config["detection_mode"] == "图片上传":
        handle_image_upload(detector, config["conf_threshold"])
    elif config["detection_mode"] == "摄像头实时检测":
        handle_camera_detection(detector, config["conf_threshold"])
    elif config["detection_mode"] == "视频文件":
        handle_video_upload(detector, config["conf_threshold"])
    
    # 页脚信息
    st.markdown("---")
    st.markdown("💡 **使用说明**：")
    st.markdown("""
    1. **图片检测**：上传单张图片进行按钮检测
    2. **摄像头检测**：使用电脑摄像头进行实时检测（推荐！）
    3. **视频检测**：上传视频文件进行批量检测
    
    **注意**：上传训练好的电梯按钮检测模型(.pt文件)可以获得更好的检测效果哦～
    """)


if __name__ == "__main__":
    main()
