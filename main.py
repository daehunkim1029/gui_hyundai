import os
import sys

from script import YOLOWrapper, open_directory

from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QGroupBox, \
    QFormLayout, QComboBox, QCheckBox, QMessageBox, QLabel, QTableWidget, QSplitter, \
    QTableWidgetItem, QFileDialog, QListWidget, QHBoxLayout
from qtpy.QtCore import Qt, QCoreApplication
from qtpy.QtGui import QFont, QPixmap

from qtpy.QtCore import QThread, Signal

os.environ['QT_API'] = 'pyqt6'

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # HighDPI support

QApplication.setFont(QFont('Arial', 12))


class Thread(QThread):
    errorGenerated = Signal(str)
    generateFinished = Signal(str, dict)

    def __init__(self, wrapper: YOLOWrapper, cur_task, path, plot_arg):
        super(Thread, self).__init__()
        self.__wrapper = wrapper
        self.__cur_task = cur_task
        self.__path = path
        self.__plot_arg = plot_arg

    def run(self):
        try:
            if os.path.exists(self.__path):
                dst_filename, result_dict = self.__wrapper.get_result(self.__cur_task, self.__path, self.__plot_arg)
                self.generateFinished.emit(dst_filename, result_dict)
            else:
                raise Exception(f'The file {self.__path} doesn\'t exists')
        except Exception as e:
            self.errorGenerated.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        self.__wrapper = YOLOWrapper()
        self.__currentIndex = -1
        self.__fileList = []

    def __initUi(self):
        self.setWindowTitle('PyQt Ultralytics YOLO GUI')

        self.__btn = QPushButton('Run')
        self.__btn.clicked.connect(self.__run)

        self.__dirBtn = QPushButton('Select Directory')
        self.__dirBtn.clicked.connect(self.__selectDirectory)

        self.__fileListWidget = QListWidget()
        self.__fileListWidget.itemSelectionChanged.connect(self.__fileSelectionChanged)

        self.__taskCmbBox = QComboBox()
        self.__taskCmbBox.addItems(['Object Detection', 'Semantic Segmentation', 'Object Tracking'])

        self.__boxesChkBox = QCheckBox()
        self.__labelsChkBox = QCheckBox()
        self.__confChkBox = QCheckBox()

        self.__boxesChkBox.setChecked(True)
        self.__labelsChkBox.setChecked(True)
        self.__confChkBox.setChecked(True)

        lay = QFormLayout()
        lay.addRow('Task', self.__taskCmbBox)
        lay.addRow('Show Boxes', self.__boxesChkBox)
        lay.addRow('Show Labels', self.__labelsChkBox)
        lay.addRow('Show Confidence', self.__confChkBox)

        settingsGrpBox = QGroupBox()
        settingsGrpBox.setTitle('Settings')
        settingsGrpBox.setLayout(lay)

        self.__prevBtn = QPushButton('Previous')
        self.__prevBtn.clicked.connect(self.__prevImage)
        self.__nextBtn = QPushButton('Next')
        self.__nextBtn.clicked.connect(self.__nextImage)

        navLay = QHBoxLayout()
        navLay.addWidget(self.__prevBtn)
        navLay.addWidget(self.__nextBtn)

        self.__imageLabel = QLabel()
        self.__imageLabel.setAlignment(Qt.AlignCenter)
        self.__imageLabel.setFixedSize(500, 500)

        lay = QVBoxLayout()
        lay.addWidget(self.__dirBtn)
        lay.addWidget(self.__fileListWidget)
        lay.addLayout(navLay)
        lay.addWidget(self.__imageLabel)
        lay.addWidget(self.__btn)
        lay.addWidget(settingsGrpBox)
        lay.setAlignment(Qt.AlignTop)

        leftWidget = QWidget()
        leftWidget.setLayout(lay)

        self.__resultImageLabel = QLabel()
        self.__resultImageLabel.setAlignment(Qt.AlignCenter)
        self.__resultImageLabel.setFixedSize(500, 500)

        self.__resultTableWidget = QTableWidget()
        self.__resultTableWidget.setEditTriggers(QTableWidget.NoEditTriggers)

        lay = QVBoxLayout()
        lay.addWidget(QLabel('Result'))
        lay.addWidget(self.__resultImageLabel)
        lay.addWidget(self.__resultTableWidget)

        rightWidget = QWidget()
        rightWidget.setLayout(lay)

        splitter = QSplitter()
        splitter.addWidget(leftWidget)
        splitter.addWidget(rightWidget)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([500, 500])
        splitter.setStyleSheet(
            "QSplitterHandle {background-color: lightgray;}")

        self.setCentralWidget(splitter)

        self.__btn.setEnabled(False)
        self.__prevBtn.setEnabled(False)
        self.__nextBtn.setEnabled(False)

    def __selectDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if dir_path:
            self.__fileListWidget.clear()
            self.__fileList = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
            self.__fileListWidget.addItems(self.__fileList)
            self.__currentIndex = 0
            self.__updateNavigationButtons()
            if self.__fileList:
                self.__displayImage(self.__fileList[self.__currentIndex])
                self.__btn.setEnabled(True)

    def __fileSelectionChanged(self):
        selected_items = self.__fileListWidget.selectedItems()
        if selected_items:
            self.__currentIndex = self.__fileList.index(selected_items[0].text())
            self.__updateNavigationButtons()
            self.__displayImage(self.__fileList[self.__currentIndex])

    def __prevImage(self):
        if self.__currentIndex > 0:
            self.__currentIndex -= 1
            self.__updateNavigationButtons()
            self.__fileListWidget.setCurrentRow(self.__currentIndex)
            self.__displayImage(self.__fileList[self.__currentIndex])

    def __nextImage(self):
        if self.__currentIndex < len(self.__fileList) - 1:
            self.__currentIndex += 1
            self.__updateNavigationButtons()
            self.__fileListWidget.setCurrentRow(self.__currentIndex)
            self.__displayImage(self.__fileList[self.__currentIndex])

    def __updateNavigationButtons(self):
        self.__prevBtn.setEnabled(self.__currentIndex > 0)
        self.__nextBtn.setEnabled(self.__currentIndex < len(self.__fileList) - 1)

    def __displayImage(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.__imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.__imageLabel.setPixmap(pixmap)

    def __run(self):
        if self.__currentIndex == -1:
            QMessageBox.warning(self, 'Warning', 'No file selected')
            return

        src_pathname = self.__fileList[self.__currentIndex]
        cur_task = self.__taskCmbBox.currentIndex()

        is_boxes_checked = self.__boxesChkBox.isChecked()
        is_labels_checked = self.__labelsChkBox.isChecked()
        is_conf_checked = self.__confChkBox.isChecked()

        plot_arg = {
            'boxes': is_boxes_checked,
            'labels': is_labels_checked,
            'conf': is_conf_checked
        }

        self.__t = Thread(self.__wrapper, cur_task, src_pathname, plot_arg)
        self.__t.started.connect(self.__started)
        self.__t.errorGenerated.connect(self.__errorGenerated)
        self.__t.generateFinished.connect(self.__generatedFinished)
        self.__t.finished.connect(self.__finished)
        self.__t.start()

    def __toggleWidget(self, f):
        self.__boxesChkBox.setEnabled(f)
        self.__labelsChkBox.setEnabled(f)
        self.__confChkBox.setEnabled(f)
        self.__btn.setEnabled(f)

    def __started(self):
        self.__toggleWidget(False)

    def __errorGenerated(self, e):
        QMessageBox.critical(self, 'Error', e)

    def __initTable(self, result_dict):
        self.__resultTableWidget.clearContents()
        self.__resultTableWidget.setRowCount(1)
        self.__resultTableWidget.setVerticalHeaderLabels(['Count'])
        self.__resultTableWidget.setColumnCount(len(result_dict))
        self.__resultTableWidget.setHorizontalHeaderLabels(list(result_dict.keys()))
        self.__resultTableWidget.setRowCount(1)
        for i, (k, v) in enumerate(result_dict.items()):
            self.__resultTableWidget.setItem(0, i, QTableWidgetItem(str(v)))

    def __generatedFinished(self, filename, result_dict):
        open_directory(os.path.dirname(filename))
        self.__initTable(result_dict)
        self.__displayResultImage(filename)

    def __displayResultImage(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.__resultImageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.__resultImageLabel.setPixmap(pixmap)

    def __finished(self):
        self.__toggleWidget(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())