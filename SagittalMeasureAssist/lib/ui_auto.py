import ctk
import qt


class AutoUI:
    """ONNX inference UI section."""

    def __init__(self, parentLayout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Auto Inference (ONNX)"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        self.modelPathEdit = qt.QLineEdit()
        self.modelBrowseButton = qt.QPushButton("Browse...")
        modelLayout = qt.QHBoxLayout()
        modelLayout.addWidget(self.modelPathEdit, 1)
        modelLayout.addWidget(self.modelBrowseButton)
        form.addRow("ONNX model:", modelLayout)

        self.runButton = qt.QPushButton("Run inference")
        form.addRow(self.runButton)

        self.heatmapCheckBox = qt.QCheckBox("Show heatmap")
        self.heatmapCheckBox.setChecked(True)
        self.heatmapCheckBox.setEnabled(False)
        form.addRow(self.heatmapCheckBox)

        self.landmarkCombo = qt.QComboBox()
        self.landmarkCombo.addItem("Composite")
        for key in ["L1_ant", "L1_post", "S1_ant", "S1_post", "FH"]:
            self.landmarkCombo.addItem(key)
        self.landmarkCombo.setEnabled(False)
        form.addRow("Landmark:", self.landmarkCombo)

        opacityLayout = qt.QHBoxLayout()
        self.opacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(50)
        self.opacitySlider.setEnabled(False)
        self.opacityLabel = qt.QLabel("50%")
        opacityLayout.addWidget(self.opacitySlider)
        opacityLayout.addWidget(self.opacityLabel)
        form.addRow("Heatmap opacity:", opacityLayout)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        form.addRow(self.statusLabel)
