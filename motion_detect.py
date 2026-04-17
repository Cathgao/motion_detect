import subprocess
import sys
import threading
import queue
import logging

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

# --- GPU运动检测配置 ---
GPU_DOWNSCALE = 1           # 缩放因子(越大越快,建议4-8)，最终处理尺寸为原图/downscale
GPU_THRESHOLD = 60          # 运动检测阈值(1-255,越小越灵敏)
HISTORY_FRAMES = 1         # 背景建模历史帧数，越大越稳定但适应慢

def get_video_size(video_path):
    """获取视频原始尺寸"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', video_path]
    res = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return map(int, res.split('x'))

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
        self.backSub = cv2.createBackgroundSubtractorKNN(history=HISTORY_FRAMES, dist2Threshold=400.0, detectShadows=True)
        
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
        '-re',
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
    finally:
        frame_queue.put((None, None))


def start_realtime_preview(frame_queue, stop_event, width, height):
    detector = OpenCLMotionDetector(width, height, threshold=GPU_THRESHOLD) 
    cv2.namedWindow("CCTV Real-time Detection", cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        try:
            # 1. 尝试从队列获取帧（带超时，防止死锁）
            idx, frame = frame_queue.get(timeout=0.02)
            
            if idx is None:  # 收到结束信号
                break
                
            # 2. 运动检测处理
            thresh_umat, motion_ratio = detector.detect_visual(frame, idx)
            thresh_cpu = thresh_umat.get()
            
            # 3. 绘图
            contours, _ = cv2.findContours(thresh_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display_frame = frame.copy() 
            for cnt in contours:
                if cv2.contourArea(cnt) < 300:
                    continue
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 4. 显示
            thresh_bgr = cv2.cvtColor(thresh_cpu, cv2.COLOR_GRAY2BGR)
            combined_view = np.hstack((display_frame[:,:,:3], thresh_bgr))
            cv2.imshow("CCTV Real-time Detection", combined_view)
            
            # 5. 告诉队列处理完了
            frame_queue.task_done()

        except queue.Empty:
            pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

def main_preview(video_path):
    # 1. 获取视频分辨率
    # ... 
    try:
        width, height = get_video_size(video_path)
    except:
        width, height = 1920, 1080 # 保底值
    global GPU_DOWNSCALE
    if(GPU_DOWNSCALE < 1):
            GPU_DOWNSCALE = 1
    scaled_w = width // GPU_DOWNSCALE
    scaled_h = height // GPU_DOWNSCALE
    logger.info(f"原始尺寸: {width}x{height}, 处理尺寸: {scaled_w}x{scaled_h}")
    
    frame_queue = queue.Queue(maxsize=128)
    stop_event = threading.Event()
    
    # 2. 启动 GPU 解码线程
    decoder_thread = threading.Thread(
        target=decode_frames_gpu, 
        args=(video_path, frame_queue, stop_event, scaled_w, scaled_h)
    )
    
    decoder_thread.start()
    
    # 3. 启动实时预览
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
        input_path = "./"
    main_preview(input_path)
        
