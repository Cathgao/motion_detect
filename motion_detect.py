import os
import subprocess
import re
import sys
import threading
import queue
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局停止事件，用于 Ctrl+C 终止
stop_event = threading.Event()

# --- 用户配置区 ---
INPUT_DIR = "./"  # 基础目录，包含多个YYYYMMDDHH格式的子文件夹
FFMPEG_PATH = "ffmpeg"  # 如果未在PATH中，填写完整路径，如 "C:/ffmpeg/bin/ffmpeg.exe"

# --- GPU运动检测配置 ---
GPU_DOWNSCALE = 4           # 缩放因子(越大越快,建议4-8)，最终处理尺寸为原图/downscale
GPU_THRESHOLD = 30          # 运动检测阈值(1-255,越小越灵敏)
GPU_SENSITIVITY = 0.0006    # 检测灵敏度(0-1,越小越灵敏)


def is_yyyymmddhh_folder(name):
    """检查文件夹名是否为YYYYMMDDHH格式（10位数字，如2026030900）"""
    return bool(re.fullmatch(r'\d{10}', name))

def is_video_file(filename):
    """检查是否为指定格式的视频文件（--M--S_timestamp.mp4）"""
    return bool(re.fullmatch(r'\d+M\d+S_\d+\.mp4', filename, re.IGNORECASE))

def get_duration(filename):
    """获取视频总时长（秒）"""
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {filename}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def merge_videos(file_list, output_path):
    """合并视频到指定路径"""
    folder_dir = os.path.dirname(output_path)
    with open(os.path.join(folder_dir, "concat_list.txt"), "w") as f:
        for file in file_list:
            f.write(f"file '{file}'\n")
    
    cmd = (
        f"{FFMPEG_PATH} -y -f concat -safe 0 -i \"{os.path.join(folder_dir, 'concat_list.txt')}\" "
        f"-fflags +genpts+igndts "
        f"-c copy -avoid_negative_ts make_zero \"{output_path}\""
    )
    subprocess.run(cmd, shell=True)

# ============= GPU加速运动检测相关函数 =============

class OpenCLMotionDetector:
    """使用 MOG2 背景建模的运动检测"""
    
    def __init__(self, width, height, threshold=30):
        self.width = width
        self.height = height
        self.threshold = threshold
        
        # 初始化 MOG2 建模器
        # history: 训练背景的帧数（默认500）
        # varThreshold: 方差阈值，类似原来的 GPU_THRESHOLD（通常16-40）
        # detectShadows: 是否检测阴影（如果设为 True，影子会被标记为灰色像素值127）
        # self.backSub = cv2.createBackgroundSubtractorMOG2(
        #     history=500, 
        #     varThreshold=threshold, 
        #     detectShadows=True
        # )
        self.backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)
        
        self.kernel_3x3 = np.ones((3, 3), np.uint8)
        
        if CV2_AVAILABLE:
            cv2.ocl.setUseOpenCL(True)

    def detect_visual(self, frame, idx):
        frame_umat = cv2.UMat(frame)
        gray = cv2.cvtColor(frame_umat, cv2.COLOR_BGRA2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        
        lr = 1.0 if idx == 0 else -1
        fg_mask = self.backSub.apply(gray, learningRate=lr)
        
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, self.kernel_3x3, iterations=1)
        thresh = cv2.dilate(thresh, self.kernel_3x3, iterations=2)
        
        # 返回 thresh 用于可视化，moving_pixels 用于逻辑判断
        moving_pixels = cv2.countNonZero(thresh)
        return thresh, moving_pixels / (self.width * self.height)


def decode_frames_gpu(video_path, frame_queue, stop_event, scaled_w, scaled_h):
    """
    GPU解码线程 - 使用FFmpeg硬件加速
    """
    cmd = [
        'ffmpeg',
        '-init_hw_device', 'qsv=qsvhw,child_device_type=dxva2',
        '-filter_hw_device', 'qsvhw',
        '-hwaccel', 'qsv',
        '-hwaccel_device', 'qsvhw',
        '-hwaccel_output_format', 'qsv',
        '-i', video_path,
        '-vf', f'vpp_qsv=w={scaled_w}:h={scaled_h}:scale_mode=hq:denoise=30:framerate=20:format=bgra,hwdownload,format=bgra',
        '-f', 'rawvideo',
        # '-pix_fmt', 'bgra',
        '-'
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        bytes_per_frame = scaled_w * scaled_h * 4
        frame_idx = 0
        
        # 单独线程读取stderr
        def read_stderr():
            for line in process.stderr:
                logger.info(f"[FFmpeg] {line.decode().rstrip()}")
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        while not stop_event.is_set():
            raw = process.stdout.read(bytes_per_frame)
            if len(raw) < bytes_per_frame:
                break
            
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((scaled_h, scaled_w, 4))
            frame_queue.put((frame_idx, frame))
            frame_idx += 1
        
        process.terminate()
        logger.info(f"解码完成: {frame_idx}帧")
    except Exception as e:
        logger.error(f"解码失败: {e}")


def process_with_gpu(frame_queue, motions, lock, stop_event, downscale, width, height, motion_threshold=30):
    """使用 GPU 处理运动检测"""
    if not CV2_AVAILABLE:
        logger.error("OpenCV 不可用，无法使用 GPU 加速")
        return
    
    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.haveOpenCL():
        logger.info(f"使用 OpenCL: {cv2.ocl.Device.getDefault().name()}")
    else:
        logger.warning("OpenCL 不可用，使用 CPU")
    
    detector = OpenCLMotionDetector(downscale, downscale, threshold=motion_threshold)
    
    count = 0
    while not stop_event.is_set():
        try:
            idx, frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        
        motion = detector.detect(frame, idx)
        
        with lock:
            motions.append((idx, motion))
            count += 1
        
        if count % 1000 == 0:
            logger.info(f"处理: {count}帧")
        
        frame_queue.task_done()


def detect_motion_gpu(video_path, heatmap_output_path, downscale=4, motion_threshold=30, sensitivity=0.0005):
    """GPU加速的运动检测，直接生成热力图 PNG
    返回: (timestamps, activity_map) 或 (None, None) 如果失败
    """
    # 重置全局停止事件
    stop_event.clear()
    
    if not CV2_AVAILABLE:
        logger.error("OpenCV 不可用，无法使用 GPU 加速运动检测")
        raise RuntimeError("需要 OpenCV 才能使用 GPU 加速运动检测功能")
    
    # 获取视频信息
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,nb_frames',
           '-of', 'json', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    width = int(stream.get('width', 1920))
    height = int(stream.get('height', 1080))
    
    # 计算缩放后的尺寸
    scaled_w = width // downscale
    scaled_h = height // downscale
    
    logger.info(f"视频: {width}x{height}, 缩放: {scaled_w}x{scaled_h}")
    
    # 线程通信
    frame_queue = queue.Queue(maxsize=1024)
    motions = []
    lock = threading.Lock()
    
    # 启动线程
    t1 = threading.Thread(target=decode_frames_gpu, args=(video_path, frame_queue, stop_event, scaled_w, scaled_h))
    t2 = threading.Thread(target=process_with_gpu, args=(frame_queue, motions, lock, stop_event, scaled_w, scaled_h, motion_threshold))
    
    t1.start()
    t2.start()
    
    # 等待完成
    t1.join()
    frame_queue.join()
    stop_event.set()
    t2.join()
    
    logger.info(f"GPU处理完成: {len(motions)}帧")
    
    # 排序
    motions.sort(key=lambda x: x[0])
    
    # ========== 生成时间轴热力图 (与motion_heatmap_gpu.py一致) ==========
    timeline = np.zeros((20, width, 3), dtype=np.uint8)
    timeline[:, :] = [128, 128, 128]  # 灰色背景
    
    total_frames = len(motions)
    for idx, motion in motions:
        x = int(idx / total_frames * width)
        x = min(x, width - 1)
        if motion > sensitivity:
            t = min(motion * 30, 1.0)
            timeline[:, x] = [0, int(255*(1-t)), 255]  # 蓝青色热力图
    
    # 保存热力图
    cv2.imwrite(heatmap_output_path, timeline)
    logger.info(f"热力图已保存: {heatmap_output_path}")
    
    # 返回None表示热力图已直接生成，不再需要generate_heatmap_graphic
    return None, None

def start_realtime_preview(frame_queue, stop_event, width, height, sensitivity=0.0006):
    detector = OpenCLMotionDetector(width, height)
    
    cv2.namedWindow("CCTV Real-time Detection", cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        try:
            # 获取一帧，设定较短的 timeout 防止卡死
            idx, frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # 1. 运行检测算法
        thresh_umat, motion_ratio = detector.detect_visual(frame, idx)
        
        # 2. 将 thresh 转换回 CPU 格式用于轮廓查找
        thresh_cpu = thresh_umat.get()
        
        # 3. 查找轮廓并绘制红框
        # findContours 在 360p 下非常快
        contours, _ = cv2.findContours(thresh_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在原图上画框。注意：frame 是 BGRA 格式，可以直接画
        display_frame = frame.copy() 
        for cnt in contours:
            if cv2.contourArea(cnt) < 500: # 忽略太小的杂点
                continue
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # 画红框
        
        # 4. 显示画面
        # 我们可以把原图和黑白掩码拼在一起显示，更有“极客感”
        thresh_bgr = cv2.cvtColor(thresh_cpu, cv2.COLOR_GRAY2BGR)
        # 将 BGRA 转为 BGR 方便拼接（或者直接截取前三通道）
        combined_view = np.hstack((display_frame[:,:,:3], thresh_bgr))
        
        # 打印当前运动占比
        cv2.putText(combined_view, f"Motion: {motion_ratio:.4f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("CCTV Real-time Detection", combined_view)
        
        # 5. 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
            
        frame_queue.task_done()

    cv2.destroyAllWindows()

def main_preview(video_path):
    # 1. 获取视频分辨率 (假设通过 ffprobe 已经拿到 width, height)
    # ... 
    scaled_w, scaled_h = 640, 360
    
    frame_queue = queue.Queue(maxsize=128)
    stop_event = threading.Event()
    
    # 2. 启动 GPU 解码线程 (保持你那个 QSV + framerate=20 的版本)
    decoder_thread = threading.Thread(
        target=decode_frames_gpu, 
        args=(video_path, frame_queue, stop_event, scaled_w, scaled_h)
    )
    
    decoder_thread.start()
    
    # 3. 启动实时预览（在主线程运行，因为 GUI 通常需要在主线程更新）
    try:
        start_realtime_preview(frame_queue, stop_event, scaled_w, scaled_h)
    except KeyboardInterrupt:
        stop_event.set()
    
    decoder_thread.join()

if __name__ == "__main__":
    import signal
    
    # 处理Ctrl+C
    def signal_handler(sig, frame):
        logger.info("正在终止...")
        stop_event.set()
    
    if sys.platform == 'win32':
        # Windows需要特殊处理
        def console_ctrl_handler(dwCtrlType):
            if dwCtrlType == 0:  # CTRL_C_EVENT
                stop_event.set()
                return True
            return False
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCtrlHandler(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(console_ctrl_handler), True)
        except:
            pass
    else:
        signal.signal(signal.SIGINT, signal_handler)
    
    # 支持命令行参数传入目标路径
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = INPUT_DIR
    main_preview(input_path)
        
