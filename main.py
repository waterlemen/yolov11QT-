from PySide6 import QtWidgets, QtCore, QtGui
import cv2
import time
from threading import Thread, Lock
import queue
import os
from datetime import datetime

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

class MWindow(QtWidgets.QMainWindow):
    updateImageSignal = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.setupUI()

        self.camBtn.clicked.connect(self.startCamera)
        self.fileBtn.clicked.connect(self.openVideoFile)
        self.detectFileBtn.clicked.connect(self.detectVideoFile)
        self.stopBtn.clicked.connect(self.stop)
        self.recBtn.clicked.connect(self.toggleRecording)

        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        self.cap = None
        self.frameQueue = queue.Queue(maxsize=1)
        self.video_writer_lock = Lock()
        self.video_source = None

        self.model = YOLO('yolo11n.pt')

        self.running = True
        self.last_time = time.time()
        self.recording = False
        self.video_writer = None
        self.recorded_frames = 0

        self.updateImageSignal.connect(self.update_treated_label)

        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('YOLOv11 实时检测系统')

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        videoLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = self.create_video_label()
        self.label_treated = self.create_video_label()
        videoLayout.addWidget(self.label_ori_video)
        videoLayout.addWidget(self.label_treated)
        mainLayout.addLayout(videoLayout)

        controlGroup = QtWidgets.QGroupBox("控制面板")
        controlLayout = QtWidgets.QHBoxLayout(controlGroup)

        self.textLog = QtWidgets.QTextBrowser()
        self.textLog.setMinimumHeight(150)
        controlLayout.addWidget(self.textLog)

        btnLayout = QtWidgets.QVBoxLayout()
        self.camBtn = QtWidgets.QPushButton('📹 打开摄像头')
        self.fileBtn = QtWidgets.QPushButton('📁 打开视频文件')
        self.detectFileBtn = QtWidgets.QPushButton('🎬 检测视频文件')
        self.stopBtn = QtWidgets.QPushButton('🛑 停止')
        self.recBtn = QtWidgets.QPushButton('⏺️ 开始录制')

        for btn in [self.camBtn, self.fileBtn, self.detectFileBtn, self.stopBtn, self.recBtn]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("font-size: 14px;")

        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.fileBtn)
        btnLayout.addWidget(self.detectFileBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(self.recBtn)
        controlLayout.addLayout(btnLayout)

        mainLayout.addWidget(controlGroup)

    def create_video_label(self):
        label = QtWidgets.QLabel()
        label.setMinimumSize(520, 400)
        label.setStyleSheet('border: 2px solid #D7E2F9;')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setScaledContents(True)
        return label

    def startCamera(self):
        if self.cap and self.cap.isOpened():
            self.textLog.append("摄像头已打开")
            return
        self.stop()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "错误", "摄像头打开失败")
            self.textLog.append("摄像头打开失败")
            return
        self.video_source = 'camera'
        self.textLog.append("摄像头打开成功")
        self.timer_camera.start(30)

    def openVideoFile(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if not file_path:
            return
        self.stop()
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "错误", "视频文件打开失败")
            self.textLog.append("视频文件打开失败")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_source = 'file'
        self.textLog.append(f"已加载视频文件: {os.path.basename(file_path)}")
        timer_interval = int(1000/fps) if fps > 0 else 30
        self.timer_camera.start(timer_interval)

    def detectVideoFile(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择要检测的视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if not file_path:
            return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self.textLog.append("❌ 无法打开视频文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs("detected_videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"detected_videos/detected_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        self.textLog.append(f"🚀 开始处理视频：{os.path.basename(file_path)}")
        self.textLog.append(f"输出路径：{out_path}")

        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)[0]
            img = results.plot(line_width=1)
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(bgr_img)
            processed += 1
            if processed % 30 == 0:
                self.textLog.append(f"已处理 {processed}/{total_frames} 帧")
            QtWidgets.QApplication.processEvents()

        cap.release()
        out.release()
        self.textLog.append("✅ 视频检测完成并保存")

    def show_camera(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            if self.video_source == 'file':
                self.textLog.append("视频播放结束")
                self.stop()
            return
        frame = cv2.resize(frame, (520, 400))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format_RGB888)
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))
        if self.frameQueue.empty():
            self.frameQueue.put(frame_rgb)

    def frameAnalyzeThreadFunc(self):
        while self.running:
            try:
                frame = self.frameQueue.get(timeout=0.1)
            except queue.Empty:
                continue
            results = self.model(frame)[0]
            img = results.plot(line_width=1)
            now = time.time()
            fps = 1.0 / (now - self.last_time) if now != self.last_time else 0
            self.last_time = now
            info_lines = [f"{results.names[int(box.cls[0].item())]}: {box.conf[0].item():.2f}" for box in results.boxes]
            self.update_log(f"FPS: {fps:.2f} - {' | '.join(info_lines) if info_lines else '未检测到目标'}")
            if not img.flags['C_CONTIGUOUS']:
                img = img.copy()
            if self.recording:
                with self.video_writer_lock:
                    if self.video_writer is None:
                        self.start_video_writer()
                    if self.video_writer is not None:
                        bgr_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        self.video_writer.write(bgr_frame)
                        self.recorded_frames += 1
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            self.updateImageSignal.emit(qImage)
            time.sleep(0.01)

    def update_log(self, message):
        QtCore.QMetaObject.invokeMethod(self.textLog, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, message))

    def start_video_writer(self):
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"recordings/recording_{timestamp}.mp4"
        for codec in ['mp4v', 'avc1', 'X264']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(save_path, fourcc, 20.0, (520, 400))
            if self.video_writer.isOpened():
                self.update_log(f"使用 {codec} 编码器开始录制: {save_path}")
                return
        self.update_log("无法初始化视频录制器，尝试的编码器都失败")
        self.video_writer = None
        self.recording = False
        self.recBtn.setText('⏺️ 开始录制')

    def toggleRecording(self):
        if not self.cap or not self.cap.isOpened():
            self.update_log("视频源未打开，无法录制")
            return
        if not self.recording:
            self.recording = True
            self.recorded_frames = 0
            self.recBtn.setText('⏹️ 停止录制')
            self.update_log("准备开始录制视频...")
        else:
            self.recording = False
            with self.video_writer_lock:
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
            self.update_log(f"视频录制已停止，共录制 {self.recorded_frames} 帧")
            self.recBtn.setText('⏺️ 开始录制')

    @QtCore.Slot(QtGui.QImage)
    def update_treated_label(self, qImage):
        self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def stop(self):
        self.timer_camera.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.update_log("视频源已关闭")
        self.label_ori_video.clear()
        self.label_treated.clear()
        if self.recording:
            self.recording = False
            with self.video_writer_lock:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
            self.recBtn.setText('⏺️ 开始录制')
            self.update_log("视频录制已停止")

    def closeEvent(self, event):
        self.running = False
        self.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication()
    window = MWindow()
    window.show()
    app.exec()