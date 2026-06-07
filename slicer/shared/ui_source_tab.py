import qt
import slicer


class SourceTabWidget:
    """DICOM / Dataset タブを構築する共有UIコンポーネント。

    各部位の MeasureUI から利用する。
    属性はすべて直接アクセス可能（volumeSelector, datasetBrowseButton 等）。
    """

    def __init__(self):
        self.sourceTabWidget = qt.QTabWidget()

        # Tab 0: DICOM
        dicom_widget = qt.QWidget()
        dicom_layout = qt.QHBoxLayout(dicom_widget)
        dicom_layout.setContentsMargins(2, 4, 2, 4)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.showHidden = False
        self.volumeSelector.showChildNodeTypes = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Select the target lateral spine X-ray volume.")

        self.prevVolumeButton = qt.QPushButton("◀")
        self.prevVolumeButton.setFixedWidth(30)
        self.prevVolumeButton.toolTip = "Switch to previous volume"
        self.nextVolumeButton = qt.QPushButton("▶")
        self.nextVolumeButton.setFixedWidth(30)
        self.nextVolumeButton.toolTip = "Switch to next volume"

        dicom_layout.addWidget(self.volumeSelector, 1)
        dicom_layout.addWidget(self.prevVolumeButton)
        dicom_layout.addWidget(self.nextVolumeButton)
        self.sourceTabWidget.addTab(dicom_widget, "DICOM")

        # Tab 1: Dataset (npy + JSON)
        ds_widget = qt.QWidget()
        ds_vlayout = qt.QVBoxLayout(ds_widget)
        ds_vlayout.setContentsMargins(2, 4, 2, 4)
        ds_vlayout.setSpacing(4)

        dir_row = qt.QHBoxLayout()
        self.datasetDirEdit = qt.QLineEdit()
        self.datasetDirEdit.setPlaceholderText("dataset ディレクトリ")
        self.datasetDirEdit.setReadOnly(True)
        self.datasetBrowseButton = qt.QPushButton("Browse…")
        self.datasetBrowseButton.setFixedWidth(70)
        dir_row.addWidget(self.datasetDirEdit, 1)
        dir_row.addWidget(self.datasetBrowseButton)

        nav_row = qt.QHBoxLayout()
        self.datasetPrevButton = qt.QPushButton("◀")
        self.datasetPrevButton.setFixedWidth(30)
        self.datasetCaseCombo = qt.QComboBox()
        self.datasetCaseCombo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.datasetNextButton = qt.QPushButton("▶")
        self.datasetNextButton.setFixedWidth(30)
        nav_row.addWidget(self.datasetPrevButton)
        nav_row.addWidget(self.datasetCaseCombo)
        nav_row.addWidget(self.datasetNextButton)

        ds_vlayout.addLayout(dir_row)
        ds_vlayout.addLayout(nav_row)
        self.sourceTabWidget.addTab(ds_widget, "Dataset")
