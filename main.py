import os
import sys
import time
import cv2
import numpy as np
from script import MMSegWrapper, open_directory
import pyqtgraph as pg
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QGroupBox, \
    QFormLayout, QCheckBox, QMessageBox, QLabel, QTableWidget, QSplitter, \
    QTableWidgetItem, QFileDialog, QListWidget, QHBoxLayout, QSizePolicy, QSpacerItem
from qtpy.QtCore import Qt, QCoreApplication, QTimer, QSize , QThread, Signal
from qtpy.QtGui import QFont, QPixmap, QImage, QColor, QPainter, QPen, QKeySequence
from PyQt5.QtGui import QFontDatabase
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QHeaderView
from qtpy.QtWidgets import QSlider
import json
import csv
from queue import Queue
import threading

os.environ['QT_API'] = 'pyqt6'

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # HighDPI support

QApplication.setFont(QFont('Hyundai Sans Head Office', 12))


def print_available_fonts():
    app = QApplication(sys.argv)
    font_db = QFontDatabase()
    font_families = font_db.families()
    
    print("Available fonts:")
    for font in font_families:
        print(font)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.__initVal()
        self.__initUi()
        self.__video_finished = False
        self.__current_frame = 0

    def __initVal(self):
        self.__wrapper = MMSegWrapper()
        self.__currentIndex = -1
        self.__fileList = []
        self.__videoCapture = None
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.__playVideo)
        self.__playing = False
        self.__showGrid = False
        self.__graph_data = {'time': [], 'abnormal_ratio': [], 'frame_number': []}
        self.__frame_count = 0
        self.__video_fps = None
        self.__current_frame = 0
        self.__frame_data = []
        self.__liveVideoCapture = None
        self.__liveTimer = QTimer(self)
        self.__liveTimer.timeout.connect(self.__playVideo)
        self.__real_time_mode = False
        self.__live_fps = 30
        self.__process_live = False 
        self.__videoWriter = None
        self.__live_video_filename = ""
        self.__frame_queue = Queue()
        self.__stop_saving = threading.Event()
        self.__video_writer_thread = None
        
    def __initUi(self):
        self.setWindowTitle('SPA LAB & Hyundai GUI')

        self.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: white;
            }
        """)

        self.__playBtn = QPushButton(QIcon('imoge/play_arrow_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png'),'')
        self.__playBtn.clicked.connect(self.__play)
        self.__playBtn.setToolTip('Play')

        self.__pauseBtn = QPushButton(QIcon('imoge/pause_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png'), '')
        self.__pauseBtn.clicked.connect(self.__pause)
        self.__pauseBtn.setToolTip('Pause')

        self.__firstBtn = QPushButton(QIcon('imoge/skip_previous_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png'), '')
        self.__firstBtn.clicked.connect(self.__goToFirst)
        self.__firstBtn.setToolTip('Go to First Frame')

        self.__lastBtn = QPushButton(QIcon('imoge/skip_next_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png'), '')
        self.__lastBtn.clicked.connect(self.__goToLast)
        self.__lastBtn.setToolTip('Go to Last Frame')

        self.__dirBtn = QPushButton('Select Directory')
        self.__dirBtn.clicked.connect(self.__selectDirectory)

        self.__fileListWidget = QListWidget()
        self.__fileListWidget.itemSelectionChanged.connect(self.__fileSelectionChanged)
        self.__fileListWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.__fileListWidget.setFixedHeight(100)

        self.__prevBtn = QPushButton('Previous')
        self.__prevBtn.clicked.connect(self.__prevImage)
        self.__nextBtn = QPushButton('Next')
        self.__nextBtn.clicked.connect(self.__nextImage)

        self.__gridChkBox = QCheckBox('Show Grid')
        self.__gridChkBox.stateChanged.connect(self.__toggleGrid)


        self.__frameSlider = QSlider(Qt.Horizontal)
        self.__frameSlider.setEnabled(False)
        self.__frameSlider.sliderPressed.connect(self.__sliderPressed)
        self.__frameSlider.sliderReleased.connect(self.__sliderReleased)
        self.__frameSlider.valueChanged.connect(self.__frameChanged)
        self.__isSliderPressed = False
        self.__realTimeChkBox = QCheckBox('Real Time Mode')
        self.__realTimeChkBox.stateChanged.connect(self.__toggleRealTimeMode)

        navLay = QHBoxLayout()
        navLay.addWidget(self.__prevBtn)
        navLay.addWidget(self.__nextBtn)


        playControlLay = QVBoxLayout()
        playControlLay.addWidget(self.__frameSlider)


        ControlLay = QHBoxLayout()
        ControlLay.addWidget(self.__firstBtn)
        ControlLay.addWidget(self.__playBtn)
        ControlLay.addWidget(self.__pauseBtn)
        ControlLay.addWidget(self.__lastBtn)

        self.__imageLabel = QLabel()
        self.__imageLabel.setAlignment(Qt.AlignCenter)
        self.__imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.__imageLabel.setMinimumSize(400, 300)  # 최소 크기 설정

        self.__resultImageLabel = QLabel()
        self.__resultImageLabel.setAlignment(Qt.AlignCenter)
        self.__resultImageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.__resultImageLabel.setMinimumSize(400, 300)  # 최소 크기 설정

        # Legend Table 생성 및 설정
        self.__legendTableWidget = QTableWidget()
        self.__legendTableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.__legendTableWidget.setColumnCount(4)
        self.__legendTableWidget.setRowCount(3)
        self.__legendTableWidget.setHorizontalHeaderLabels(['Color', 'Description','Color', 'Description'])
        self.__populateLegendTable()

        header = self.__legendTableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(False)

        self.__legendTableWidget.verticalHeader().setVisible(False)
        self.__legendTableWidget.setShowGrid(False)
        self.__legendTableWidget.setStyleSheet("""
            QTableWidget {
                background-color: #f6f3f2;
                border: 1px solid #002c5f;
                border-radius: 5px;
                padding: 5px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #002c5f;
                color: white;
                padding: 5px;
                border: 1px solid #001f3f;
                font-weight: bold;
            }
        """)
        self.__legendTableWidget.setFixedHeight(130)


        # 결과 테이블 위젯 생성
        self.__resultTableWidget = QTableWidget()
        self.__resultTableWidget.setColumnCount(3)
        self.__resultTableWidget.setRowCount(2)
        self.__resultTableWidget.setHorizontalHeaderLabels(['Type', 'Contamination Level','Patch Count'])
        header = self.__resultTableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch) 
        header.setStretchLastSection(False)
        self.__resultTableWidget.verticalHeader().setVisible(False)
        self.__resultTableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.__resultTableWidget.setStyleSheet("""
            QTableWidget {
                background-color: #f6f3f2;
                border: 1px solid #002c5f;
                border-radius: 5px;
                padding: 5px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #002c5f;
                color: white;
                padding: 5px;
                border: 1px solid #001f3f;
                font-weight: bold;
            }
        """)
        self.__resultTableWidget.setFixedHeight(105)

        # 결과 테이블 초기화
        self.__initResultTable()

        # Legend 레이블 스타일 설정
        legend_label = QLabel('Legend')
        legend_label.setAlignment(Qt.AlignCenter)
        legend_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: white;
                background: #002c5f;
                border-radius: 5px;
                padding: 5px;
                margin-top: 15px;
                margin-bottom: 10px;
            }
        """)

        gridLayout = QHBoxLayout()
        gridLayout.addWidget(self.__gridChkBox)
        gridLayout.addWidget(self.__realTimeChkBox)
        gridLayout.addStretch()

        lay = QVBoxLayout()
        lay.addWidget(self.__dirBtn)
        lay.addWidget(self.__fileListWidget)
        lay.addLayout(gridLayout) 
        lay.addLayout(navLay)
        lay.setAlignment(Qt.AlignTop)
        lay.addWidget(self.__imageLabel)
        lay.addLayout(playControlLay)
        lay.addLayout(ControlLay)
        lay.addWidget(legend_label)
        lay.addWidget(self.__legendTableWidget)

        leftWidget = QWidget()
        leftWidget.setLayout(lay)

        self.__plot_widget = pg.PlotWidget()
        self.__plot_widget.setBackground('w')
        self.__plot_widget.setLabel('left', 'Contamination Level (%)')
        self.__plot_widget.setLabel('bottom', 'Time (s)')
        self.__plot_widget.showGrid(x=True, y=True)
        self.__plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.__plot_widget.setMinimumSize(200,150)
        self.__plot_curve = self.__plot_widget.plot(pen='#002c5f')
        self.__plot_curve_abnormal = self.__plot_widget.plot(pen=pg.mkPen(color='red', width=3))
        self.__vertical_line = pg.InfiniteLine( angle=90, movable=False, pen=pg.mkPen('g', width=1))
        self.__plot_widget.addItem(self.__vertical_line)
        self.__plot_widget.scene().sigMouseClicked.connect(self.__onPlotClicked)


        result_label = QLabel('Result')
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: white;
                background: #002c5f;
                border-radius: 5px;
                padding: 10px;
                margin-top: 0px;
                margin-bottom: 10px;
                text-align: center;
            }
        """)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(result_label)
        right_layout.addWidget(self.__resultTableWidget)  # 결과 테이블 추가
        right_layout.addWidget(self.__resultImageLabel)

        # 그래프를 담을 새로운 위젯 생성
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.addWidget(self.__plot_widget)
        graph_widget.setLayout(graph_layout)

        # 스플리터 생성 및 설정
        right_splitter = QSplitter(Qt.Vertical)
        top_widget = QWidget()
        top_widget.setLayout(right_layout)
        right_splitter.addWidget(top_widget)
        right_splitter.addWidget(graph_widget)

        # 스플리터 비율 설정
        right_splitter.setStretchFactor(0, 2)  # 상단 위젯에 더 많은 공간 할당
        right_splitter.setStretchFactor(1, 1)  # 그래프 위젯에 더 적은 공간 할당

        rightWidget = QWidget()
        rightWidget.setLayout(QVBoxLayout())
        rightWidget.layout().addWidget(right_splitter)

        splitter = QSplitter()
        splitter.addWidget(leftWidget)
        splitter.addWidget(rightWidget)
        splitter.setHandleWidth(3)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #002c5f;
                border-left: 1px solid #001f3f;
                border-right: 1px solid #001f3f;
                width: 10px;
            }
            QSplitter::handle:hover {
                background-color: #003366;
            }
        """)

        self.setCentralWidget(splitter)

        self.__playBtn.setEnabled(False)
        self.__pauseBtn.setEnabled(False)
        self.__firstBtn.setEnabled(False)
        self.__lastBtn.setEnabled(False)

        button_style = """
            QPushButton {
                background-color: #ffffff;
                color: #002c5f;
                border: 2px solid #002c5f;
                border-radius: 5px;
                padding: 5px;
                min-width: 80px;
                min-height: 30px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #002c5f;
                color: white;
            }
            QPushButton:pressed {
                background-color: #001f3f;
                border-color: #001f3f;
                color: white;
            }
            QPushButton:disabled {
                background-color: #ecf0f1;
                border-color: #bdc3c7;
                color: #7f8c8d;
            }
        """

        for btn in [self.__playBtn, self.__pauseBtn, self.__firstBtn, self.__lastBtn, self.__prevBtn, self.__nextBtn, self.__dirBtn]:
            btn.setStyleSheet(button_style)


    def __selectDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if dir_path:
            self.__fileListWidget.clear()
            self.__fileList = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
            self.__fileListWidget.addItems(self.__fileList)
            self.__currentIndex = 0
            self.__updateNavigationButtons()
            if self.__fileList:
                current_file = self.__fileList[self.__currentIndex]
                if current_file.endswith('.mp4'):
                    self.__videoCapture = cv2.VideoCapture(current_file)
                    self.__playBtn.setEnabled(True)
                    self.__pauseBtn.setEnabled(True)
                    self.__firstBtn.setEnabled(True)
                    self.__lastBtn.setEnabled(True)
                else:
                    self.__displayImage(self.__fileList[self.__currentIndex])
                    self.__playBtn.setEnabled(True)
                    self.__pauseBtn.setEnabled(True)
    
    def __fileSelectionChanged(self):
        selected_items = self.__fileListWidget.selectedItems()
        if selected_items:
            self.__currentIndex = self.__fileList.index(selected_items[0].text())
            self.__updateNavigationButtons()
            self.__displayImage(self.__fileList[self.__currentIndex])
            self.__updateVideoCapture()
            self.__resetGraph()
            self.__frame_data = []

    def __toggleGrid(self, state):
        self.__showGrid = state == Qt.Checked
        if hasattr(self, '__resultImagePath'):
            self.__displayResultImage(self.__resultImagePath)

    def __addGrid(self, pixmap):
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 170, 210), 1, Qt.SolidLine)
        painter.setPen(pen)
        rows, cols = 16, 16
        row_step = pixmap.height() / rows
        col_step = pixmap.width() / cols

        for i in range(rows + 1):
            painter.drawLine(0, int(i * row_step), pixmap.width(), int(i * row_step))
        for j in range(cols + 1):
            painter.drawLine(int(j * col_step), 0, int(j * col_step), pixmap.height())
        
        painter.end()
        return pixmap
    
    
    def __prevImage(self):
        if self.__currentIndex > 0:
            self.__currentIndex -= 1
            self.__updateNavigationButtons()
            self.__fileListWidget.setCurrentRow(self.__currentIndex)
            self.__displayImage(self.__fileList[self.__currentIndex])
            self.__updateVideoCapture()
            self.__resetGraph()
            self.__frame_data = []

    def __nextImage(self):
        if self.__currentIndex < len(self.__fileList) - 1:
            self.__currentIndex += 1
            self.__updateNavigationButtons()
            self.__fileListWidget.setCurrentRow(self.__currentIndex)
            self.__displayImage(self.__fileList[self.__currentIndex])
            self.__updateVideoCapture()
            self.__resetGraph()
            self.__frame_data = []
    
    def __displayImage(self, path):
        if path.endswith('.mp4'):
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
            else:
                QMessageBox.warning(self, "비디오 오류", f"{path} 파일을 읽을 수 없습니다.")
                pixmap = QPixmap()  # 빈 픽스맵 설정
            cap.release()
        else:
            pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.__imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.__imageLabel.setPixmap(pixmap)

    def __updateNavigationButtons(self):
        has_files = bool(self.__fileList)
        self.__prevBtn.setEnabled(self.__currentIndex > 0)
        self.__nextBtn.setEnabled(self.__currentIndex < len(self.__fileList) - 1)
        self.__playBtn.setEnabled(has_files)
        self.__pauseBtn.setEnabled(has_files)
    

    def __updateVideoCapture(self):
        current_file = self.__fileList[self.__currentIndex]
        if current_file.endswith('.mp4'):
            if self.__videoCapture is not None:
                self.__videoCapture.release()
            self.__videoCapture = cv2.VideoCapture(current_file)
            self.__playBtn.setEnabled(True)
            self.__pauseBtn.setEnabled(True)
            self.__firstBtn.setEnabled(True)
            self.__lastBtn.setEnabled(True)
            self.__frameSlider.setEnabled(True)
            
            total_frames = int(self.__videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.__frameSlider.setRange(0, total_frames - 1)
            
            # 비디오 FPS 가져오기
            self.__video_fps = self.__videoCapture.get(cv2.CAP_PROP_FPS)
            self.__video_duration = total_frames / self.__video_fps
            self.__current_frame = 0
        else:
            if self.__videoCapture is not None:
                self.__videoCapture.release()
                self.__videoCapture = None
            self.__playBtn.setEnabled(False)
            self.__pauseBtn.setEnabled(False)
            self.__firstBtn.setEnabled(False)
            self.__lastBtn.setEnabled(False)
            self.__frameSlider.setEnabled(False)
            self.__current_frame = 0
    
    def __frameChanged(self):
        if self.__videoCapture is not None and self.__isSliderPressed:
            frame_pos = self.__frameSlider.value()
            self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = self.__videoCapture.read()
            if ret:
                self.__displayFrame(frame)
                dst_filename, result_dict = self.__wrapper.get_result(frame)
                self.__displayResultImage(dst_filename)
                self.__current_frame = frame_pos  # 현재 프레임 업데이트
                self.__resetGraph()  # 그래프 초기화
                self.__frame_data = []
                self.__updateGraphData(result_dict)  # 현재 프레임의 데이터로 그래프 업데이트
                abnormal_ratio = self.__graph_data['abnormal_ratio'][-1]
                self.__updateResultTable(result_dict, abnormal_ratio)


    def __sliderPressed(self):
        self.__isSliderPressed = True

    def __sliderReleased(self):
        self.__isSliderPressed = False


    def __goToFirst(self):
        if self.__videoCapture is not None:
            self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.__videoCapture.read()
        if ret:
            self.__displayFrame(frame)
            dst_filename, _ = self.__wrapper.get_result(frame)
            self.__displayResultImage(dst_filename)
            self.__resetGraph()
            self.__frame_data = []
            self.__current_frame = 0  # 현재 프레임을 0으로 설정

    def __goToLast(self):
        if self.__videoCapture is not None:
            total_frames = int(self.__videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = self.__videoCapture.read()
            if ret:
                self.__displayFrame(frame)
                dst_filename, _ = self.__wrapper.get_result(frame)
                self.__displayResultImage(dst_filename)
                self.__resetGraph()
                self.__frame_data = []
                self.__current_frame = total_frames - 1
    
    
    def __play(self):
        if self.__real_time_mode:
            if not self.__process_live:
                self.__process_live = True
                self.__playing = True  # 추가: 재생 상태 설정
                self.__liveTimer.start(1000 // self.__live_fps)  # 타이머 시작 (프레임 속도에 맞게 조정)
        else:
            current_file = self.__fileList[self.__currentIndex]
            if current_file.endswith('.mp4'):
                if not self.__playing:
                    self.__playing = True
                    self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, self.__current_frame)
                    self.__timer.start(int(1000 / self.__video_fps))
                    if self.__frameSlider.maximum():
                        self.__realTimeChkBox.setEnabled(True)
                    else:
                        self.__realTimeChkBox.setEnabled(False)

    def __pause(self):
        if self.__playing:
            self.__playing = False
            if self.__real_time_mode:
                self.__liveTimer.stop()
                self.__process_live = False  # 처리 중지
            else:
                self.__timer.stop()
                self.__current_frame = int(self.__videoCapture.get(cv2.CAP_PROP_POS_FRAMES))


    def __playVideo(self):
        
        if self.__real_time_mode:
            self.__frameSlider.setEnabled(False)
            if self.__liveVideoCapture.isOpened():
                ret, frame = self.__liveVideoCapture.read()
                if ret:
                    self.__frame_count += 1
                    if self.__frame_count % 1 == 0: 
                        # 프레임을 RGB로 변환
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        q_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0],
                                        frame_rgb.strides[0], QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        self.__imageLabel.setPixmap(pixmap.scaled(
                            self.__imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        if self.__process_live:
                            # 인퍼런스 수행 및 결과 업데이트
                            dst_filename, result_dict = self.__wrapper.get_result(frame)
                            self.__displayResultImage(dst_filename)
                            abnormal_ratio = self.__updateGraphData(result_dict)
                            self.__updateResultTable(result_dict, abnormal_ratio)

                            # 프레임 데이터 업데이트
                            if self.__video_fps is None or self.__video_fps == 0 or self.__video_fps ==-1:
                                current_time = self.__current_frame / self.__live_fps
                            else:
                                current_time = self.__current_frame / self.__video_fps
                            self.__frame_data.append({
                                'time': current_time,
                                'contamination_level': abnormal_ratio,
                                'patch_counts': {str(val): count for val, count in zip(result_dict['unique_values'], result_dict['counts'])},
                                'patch_array': result_dict.get('seg_map', [])
                            })

                                # VideoWriter에 프레임 기록
                            if self.__videoWriter is not None:
                                self.__frame_queue.put(frame.copy())
                       

                            self.__current_frame += 1
                else:
                    QMessageBox.warning(self, "실시간 비디오", "비디오 스트림을 읽을 수 없습니다.")
                    self.__stopLiveVideoCapture()
        
        else:
            self.__realTimeChkBox.setEnabled(False)
            if self.__videoCapture.isOpened() and not self.__isSliderPressed:
                ret, frame = self.__videoCapture.read()
                if ret:
                    self.__frame_count += 1
                    if self.__frame_count % 2 == 0:  # n프레임마다 처리
                        dst_filename, result_dict = self.__wrapper.get_result(frame)
                        self.__displayFrame(frame)
                        self.__displayResultImage(dst_filename)
                        abnormal_ratio = self.__updateGraphData(result_dict)
                        self.__updateResultTable(result_dict, abnormal_ratio)

                        current_time = self.__current_frame / self.__video_fps
                        patch_counts = {str(val): count for val, count in zip(result_dict['unique_values'], result_dict['counts'])}
                        patch_array = result_dict['seg_map'] if 'seg_map' in result_dict else []
                        frame_data = {
                        'time': current_time,
                        'contamination_level': abnormal_ratio,
                        'patch_counts': patch_counts,
                        'patch_array': patch_array
                    }
                        self.__frame_data.append(frame_data)
                        #print(self.__frame_data)

                    self.__current_frame = int(self.__videoCapture.get(cv2.CAP_PROP_POS_FRAMES))
                    self.__frameSlider.setValue(self.__current_frame)

                    if self.__current_frame == self.__frameSlider.maximum():
                        self.__video_finished = True
                        self.__realTimeChkBox.setEnabled(True)
                        self.__pause()

                        # 팝업 창 표시 (Y/N 키 단축키 설정)
                        msgBox = QMessageBox(self)
                        msgBox.setWindowTitle('저장 확인')
                        msgBox.setText('저장하시겠습니까?')
                        msgBox.setIcon(QMessageBox.Question)
                        
                        # "No" 버튼 먼저 추가
                        no_button = msgBox.addButton('No', QMessageBox.NoRole)
                        # "Yes" 버튼 나중에 추가
                        yes_button = msgBox.addButton('Yes', QMessageBox.YesRole)
                        
                        # 버튼에 단축키 설정
                        no_button.setShortcut(QKeySequence(Qt.Key_N))
                        yes_button.setShortcut(QKeySequence(Qt.Key_Y))
                        
                        # 기본 버튼 설정 (필요시 설정 가능)
                        msgBox.setDefaultButton(no_button)
                        
                        # 메시지 박스 실행
                        msgBox.exec()
                        
                        # 사용자 응답 처리
                        if msgBox.clickedButton() == yes_button:
                            self.__saveResults()
                        elif msgBox.clickedButton() == no_button:
                            pass  # "No"를 선택한 경우 추가 작업이 필요하면 여기에 작성
                        self.__realTimeChkBox.setEnabled(True)
                else:
                    self.__pause()
                    #self.__video_finished = True
        

    
    def __saveResults(self):
        # 저장할 디렉토리 선택
        options = QFileDialog.Options()
        dir_name = QFileDialog.getExistingDirectory(
            self,
            "결과 저장할 디렉토리 선택",
            "",
            options=options
        )
        
        if dir_name:
            try:
                dir_name = os.path.join(dir_name, 'result')
                os.makedirs(dir_name, exist_ok=True)

                json_dir = os.path.join(dir_name, "json_result")
                txt_dir = os.path.join(dir_name, "txt_result")
                csv_dir = os.path.join(dir_name, "csv_result")
                graph_dir = os.path.join(dir_name, "graph_result")
                online_dir = os.path.join(dir_name, "online_raw_video")
                os.makedirs(json_dir, exist_ok=True)
                os.makedirs(txt_dir, exist_ok=True)
                os.makedirs(csv_dir, exist_ok=True)
                os.makedirs(graph_dir, exist_ok=True)
                os.makedirs(online_dir,exist_ok=True)
                
                if self.__real_time_mode:
                    base_name = "live_capture"
                else:
                    if self.__currentIndex < 0 or self.__currentIndex >= len(self.__fileList):
                        QMessageBox.warning(self, "저장 실패", "유효한 파일이 선택되지 않았습니다.")
                        return
                    current_file = self.__fileList[self.__currentIndex]
                    base_name = os.path.splitext(os.path.basename(current_file))[0]
                
                if not self.__frame_data:
                    QMessageBox.warning(self, "저장 실패", "저장할 데이터가 없습니다.")
                    return
                
                start_time = self.__frame_data[0]['time']
                end_time = self.__frame_data[-1]['time']
                
                def format_time(seconds):
                    hrs = int(seconds // 3600)
                    mins = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    return f"{hrs:02}:{mins:02}:{secs:02}"
                
                start_time_str = format_time(start_time)
                end_time_str = format_time(end_time)
                
                # 파일명 생성
                file_base = f"{base_name}_{start_time_str}_{end_time_str}"
                json_file = os.path.join(json_dir, f"{file_base}.json")
                txt_file = os.path.join(txt_dir, f"{file_base}.txt")
                csv_file = os.path.join(csv_dir, f"{file_base}.csv")
                graph_file = os.path.join(graph_dir, f"{file_base}.png")
                
                if self.__real_time_mode:
                    video_file = os.path.join(online_dir, f"{file_base}.mp4")
                else:
                    video_file = os.path.join(online_dir, f"{file_base}.mp4")
                
                # JSON 저장
                serializable_data = []
                for frame in self.__frame_data:
                    frame_copy = frame.copy()
                    
                    if 'patch_array' in frame_copy and isinstance(frame_copy['patch_array'], np.ndarray):
                        frame_copy['patch_array'] = frame_copy['patch_array'].tolist()
                    if 'patch_counts' in frame_copy and isinstance(frame_copy['patch_counts'], dict):
                        for key in frame_copy['patch_counts']:
                            frame_copy['patch_counts'][key] = int(frame_copy['patch_counts'][key])
                    if 'time' in frame_copy:
                        frame_copy['time'] = float(frame_copy['time'])
                    if 'contamination_level' in frame_copy:
                        frame_copy['contamination_level'] = float(frame_copy['contamination_level'])
                    
                    serializable_data.append(frame_copy)
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=4)
                
                label_map = {0: "Clean", 1: "Rain blur", 2: "Rain blockage"}

                with open(txt_file, 'w', encoding='utf-8') as f:
                    for frame in self.__frame_data:
                        time_str = format_time(frame['time'])
                        contamination = frame.get('contamination_level', 0)
                        counts = frame.get('patch_counts', {})
                        counts_str = ', '.join([f"{label_map.get(int(k), 'unknown')}: {v}" for k, v in counts.items()])
                        f.write("-------------------------------------------------------------------------------------------------------\n")
                        f.write(f"Time: {time_str}, Contamination Level: {contamination:.2%}, Patch Counts: {counts_str}\n")
                        
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 헤더 작성
                    writer.writerow(['Time (s)', 'Contamination Level (%)', 'Frame Number'])
                    # 데이터 작성
                    for t, ratio, frame_num in zip(self.__graph_data['time'], self.__graph_data['abnormal_ratio'], self.__graph_data['frame_number']):
                        writer.writerow([f"{t:.2f}", f"{ratio * 100:.2f}", frame_num])
                
                pixmap = self.__plot_widget.grab()
                pixmap.save(graph_file, 'PNG')
                
                # 실시간 모드에서 VideoWriter를 사용하여 저장된 비디오를 지정된 경로로 이동
                if self.__real_time_mode and self.__videoWriter is not None:
                    self.__videoWriter.release()
                    # 비디오 파일을 지정된 저장 디렉토리로 이동
                    os.rename(self.__live_video_filename, video_file)
                    self.__videoWriter = None  # VideoWriter 객체 초기화
                
                QMessageBox.information(self, "저장 완료", f"결과가 성공적으로 저장되었습니다:\nJSON: {json_file}\nTXT: {txt_file}\nCSV: {csv_file}\nGraph: {graph_file}\nVideo: {video_file}")
            
            except Exception as e:
                QMessageBox.warning(self, "저장 실패", f"결과 저장 중 오류가 발생했습니다:\n{str(e)}")


                

    def __onPlotClicked(self, event):
        if self.__video_finished and QApplication.keyboardModifiers() == Qt.ShiftModifier:
            pos = event.scenePos()
            if self.__plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.__plot_widget.getPlotItem().vb.mapSceneToView(pos)
                clicked_time = mouse_point.x()
                
                # Find the closest time in the graph data
                closest_time_index = min(range(len(self.__graph_data['time'])), 
                                         key=lambda i: abs(self.__graph_data['time'][i] - clicked_time))
                
                frame_number = self.__graph_data['frame_number'][closest_time_index]
                
                self.__videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = self.__videoCapture.read()
                if ret:
                    self.__displayFrame(frame)
                    dst_filename, result_dict = self.__wrapper.get_result(frame)
                    self.__displayResultImage(dst_filename)
                    self.__frameSlider.setValue(frame_number)
                    self.__vertical_line.setValue(self.__graph_data['time'][closest_time_index])
                    
                    # Update result table
                    abnormal_ratio = self.__graph_data['abnormal_ratio'][closest_time_index]
                    self.__updateResultTable(result_dict, abnormal_ratio)

    def __updateGraphData(self, result_dict):
        
        current_time = self.__current_frame / self.__video_fps
        current_frame = self.__current_frame
        
        # Check if 'counts' key exists, if not, use an alternative method to get the data
        if 'counts' in result_dict:
            total_patches = sum(result_dict['counts'])
            abnormal_patches = sum(result_dict['counts'][1:])
        else:
            # Assuming the result is a 16x16 array where non-zero values are abnormal
            print("Warning: 'counts' not found in result_dict")
            return
        
        abnormal_ratio = abnormal_patches / total_patches if total_patches > 0 else 0

        self.__graph_data['time'].append(current_time)
        self.__graph_data['abnormal_ratio'].append(abnormal_ratio)
        self.__graph_data['frame_number'].append(current_frame)

        # Limit the data points to the last 1000 for performance
        if len(self.__graph_data['time']) > 1000:
            self.__graph_data['time'] = self.__graph_data['time'][-1000:]
            self.__graph_data['abnormal_ratio'] = self.__graph_data['abnormal_ratio'][-1000:]
            self.__graph_data['frame_number'] = self.__graph_data['frame_number'][-1000:]

        # Update the plot
        if abnormal_ratio <= 0.7:
            self.__plot_curve.setData(self.__graph_data['time'], self.__graph_data['abnormal_ratio'])
        else:
            self.__plot_curve_abnormal.setData(self.__graph_data['time'], self.__graph_data['abnormal_ratio'])

        self.__vertical_line.setValue(current_time)
        
        
        return abnormal_ratio

    def __resetGraph(self):
        self.__graph_data = {'time': [], 'abnormal_ratio': [], 'frame_number': []}
        self.__plot_curve.setData([], [])
        self.__plot_curve_abnormal.setData([], [])
        self.__vertical_line.setValue(0)
        self.__video_finished = False
        self.__current_frame = 0

    
    def __displayFrame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_pixmap = QPixmap.fromImage(qt_image).scaled(self.__imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.__imageLabel.setPixmap(qt_pixmap)

    def __displayResultImage(self, img):
        if isinstance(img, str):  # img가 파일 경로일 때
            pixmap = QPixmap(img)
        elif isinstance(img, np.ndarray):  # img가 numpy 배열일 때
            if img.ndim == 2:  # grayscale image
                h, w = img.shape
                qt_image = QImage(img.data, w, h, QImage.Format_Grayscale8)
            else:  # color image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qt_image)
        else:
            raise ValueError("Unsupported image format")

        pixmap = pixmap.scaled(self.__resultImageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if self.__showGrid:
            pixmap=self.__addGrid(pixmap.copy())
        self.__resultImageLabel.setPixmap(pixmap)

    

    def __initResultTable(self):
        types = ['Rain Blur', 'Rain Block']
        for i, type_name in enumerate(types):
            # Type 열 아이템 생성 및 중앙 정렬
            type_item = QTableWidgetItem(type_name)
            type_item.setTextAlignment(Qt.AlignCenter)
            self.__resultTableWidget.setItem(i, 0, type_item)

            # Contamination Level 열 아이템 생성 및 중앙 정렬
            level_item = QTableWidgetItem('0')  # 초기값 0으로 설정
            level_item.setTextAlignment(Qt.AlignCenter)
            self.__resultTableWidget.setItem(0, 1, level_item)
            self.__resultTableWidget.setSpan(0, 1, 2, 1)

            # Count 열 아이템 생성 및 중앙 정렬 (초기값은 빈 문자열)
            count_item = QTableWidgetItem('0')
            count_item.setTextAlignment(Qt.AlignCenter)
            self.__resultTableWidget.setItem(i, 2, count_item)

        # 열 너비 조정
        #self.__resultTableWidget.resizeColumnsToContents()
    
    
    def __updateResultTable(self, result_dict,abnormal_ratio):
        if 'counts' in result_dict and 'unique_values' in result_dict:
            counts = result_dict['counts']
            unique_values = result_dict['unique_values']
            
            # value 1 (Rain Blur)과 value 2 (Rain Block)의 인덱스 찾기
            index_1 = np.where(unique_values == 1)[0]
            index_2 = np.where(unique_values == 2)[0]
            
            # Rain Blur (value 1) count 업데이트
            if len(index_1) > 0:
                count_1 = counts[index_1[0]]
                self.__resultTableWidget.item(0, 2).setText(str(count_1))
            else:
                self.__resultTableWidget.item(0, 2).setText('0')
            
            # Rain Block (value 2) count 업데이트
            if len(index_2) > 0:
                count_2 = counts[index_2[0]]
                self.__resultTableWidget.item(1, 2).setText(str(count_2))
            else:
                self.__resultTableWidget.item(1, 2).setText('0')

            contamination_level = f"{abnormal_ratio:.2%}"
            self.__resultTableWidget.item(0, 1).setText(contamination_level)

    def __populateLegendTable(self):
        colors = [
            (QColor(0, 191, 255, 128), "Rain Blur"),
            (QColor(65, 105, 225), "Rain Block"),
            (QColor(255, 165, 0, 128), "Dust Blur"),
            (QColor(255, 165, 0), "Dust Block"),
            (QColor(0, 255, 0, 128), "Snow Blur"),
            (QColor(0, 255, 0), "Snow Block")
        ]
        for i, (color, desc) in enumerate(colors):
            row = i // 2
            col = (i % 2) * 2
            color_item = QTableWidgetItem()
            color_item.setBackground(color)
            color_item.setTextAlignment(Qt.AlignCenter) 
            desc_item = QTableWidgetItem(desc)
            desc_item.setTextAlignment(Qt.AlignCenter) 
            self.__legendTableWidget.setItem(row, col, color_item)
            self.__legendTableWidget.setItem(row, col + 1, desc_item)


    def __toggleRealTimeMode(self, state):
        if state == Qt.Checked:
            self.__real_time_mode = True
            self.__process_live = False  # 초기에는 처리를 비활성화
            self.__playing = False
            self.__initLiveVideoCapture()
            self.__current_frame = 0
        else:
            self.__real_time_mode = False
            self.__process_live = False
            self.__playing = False
            self.__stopLiveVideoCapture()

    def __initLiveVideoCapture(self):
        if self.__liveVideoCapture is None:
            self.__liveVideoCapture = cv2.VideoCapture('/dev/video0')
            if not self.__liveVideoCapture.isOpened():
                QMessageBox.critical(self, "비디오 오류", "/dev/video0을 열 수 없습니다.")
                self.__realTimeChkBox.setChecked(False)
                self.__real_time_mode = False
                return
            self.__liveVideoCapture.set(cv2.CAP_PROP_FPS, self.__live_fps)
            self.__video_fps = self.__liveVideoCapture.get(cv2.CAP_PROP_FPS)
            if self.__video_fps == 0 or self.__video_fps is None or self.__video_fps == -1:
                self.__video_fps = self.__live_fps
            actual_width = int(self.__liveVideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.__liveVideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"실시간 비디오 해상도: {actual_width}x{actual_height}, FPS: {self.__video_fps}")
            self.__liveTimer.start(1000 // self.__live_fps)
            self.__playBtn.setEnabled(True)
            self.__pauseBtn.setEnabled(True)
            self.__firstBtn.setEnabled(False)
            self.__lastBtn.setEnabled(False)
            self.__prevBtn.setEnabled(False)
            self.__nextBtn.setEnabled(False)
            self.__resetGraph()
            self.__current_frame = 0
            self.__frameSlider.setEnabled(False)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.__live_video_filename = os.path.join(os.getcwd(), f"live_capture_{timestamp}.mp4")
            self.__videoWriter = cv2.VideoWriter(self.__live_video_filename, fourcc, self.__video_fps, (actual_width, actual_height))
            if not self.__videoWriter.isOpened():
                QMessageBox.critical(self, "비디오 저장 오류", "VideoWriter를 초기화할 수 없습니다.")
                self.__videoWriter = None

            # 추가된 부분 시작
            if self.__videoWriter is not None:
                self.__stop_saving.clear()
                self.__video_writer_thread = threading.Thread(target=self.__video_writer_worker)
                self.__video_writer_thread.start()
            # 추가된 부분 끝




    def __stopLiveVideoCapture(self):
        if self.__liveVideoCapture is not None:
            self.__liveVideoCapture.release()
            self.__liveVideoCapture = None
        self.__liveTimer.stop()
        self.__playing = False
        self.__process_live = False
        self.__playBtn.setEnabled(True)
        self.__pauseBtn.setEnabled(False)
        self.__imageLabel.clear()
        self.__resultImageLabel.clear()
        self.__resetGraph()
        self.__initResultTable()

        # 추가된 부분 시작
        if self.__video_writer_thread is not None:
            self.__stop_saving.set()
            self.__video_writer_thread.join()
            self.__video_writer_thread = None

        if self.__videoWriter is not None:
            self.__videoWriter.release()
            self.__videoWriter = None
        
            # VideoWriter가 활성화되어 있으면 종료
            if self.__videoWriter is not None:
                self.__videoWriter.release()
                self.__videoWriter = None

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S and self.__process_live == True:
            if self.__real_time_mode and self.__playing:
                self.__pause()  # 재생 중지
                self.__promptSaveResults()
                self.__realTimeChkBox.click()
                self.__realTimeChkBox.click()
                
        else:
            super(MainWindow, self).keyPressEvent(event)

    def __promptSaveResults(self):
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle('저장 확인')
        msgBox.setText('저장하시겠습니까?')
        msgBox.setIcon(QMessageBox.Question)
        
        # "No" 버튼 먼저 추가
        no_button = msgBox.addButton('No', QMessageBox.NoRole)
        # "Yes" 버튼 나중에 추가
        yes_button = msgBox.addButton('Yes', QMessageBox.YesRole)
        
        # 버튼에 단축키 설정
        no_button.setShortcut(QKeySequence(Qt.Key_N))
        yes_button.setShortcut(QKeySequence(Qt.Key_Y))
        
        # 기본 버튼 설정 (필요시 설정 가능)
        msgBox.setDefaultButton(no_button)
        
        # 메시지 박스 실행
        msgBox.exec()
        
        # 사용자 응답 처리
        if msgBox.clickedButton() == yes_button:
            self.__saveResults()

    def __video_writer_worker(self):
        while not self.__stop_saving.is_set() or not self.__frame_queue.empty():
            try:
                frame = self.__frame_queue.get(timeout=0.1)
                if self.__videoWriter is not None:
                    self.__videoWriter.write(frame)
                self.__frame_queue.task_done()
            except:
                continue

    def closeEvent(self, event):
        self.__stop_saving.set()
        if self.__video_writer_thread is not None:
            self.__video_writer_thread.join()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    #print_available_fonts()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())