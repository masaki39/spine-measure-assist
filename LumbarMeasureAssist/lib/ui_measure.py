import ctk
import qt
import slicer


class MeasureUI:
    """
    Builds the measurement panel (volume/markups selection, controls, results table).
    """

    def __init__(self, parentLayout, set_names=None, initial_set=None):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Measure"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        if set_names and len(set_names) > 1:
            self.setCombo = qt.QComboBox()
            for name in set_names:
                self.setCombo.addItem(name)
            if initial_set:
                self.setCombo.setCurrentText(initial_set.name)
            form.addRow("Measurement Set:", self.setCombo)
        else:
            self.setCombo = None

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.showHidden = False
        self.volumeSelector.showChildNodeTypes = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip(
            "Select the target lateral spine X-ray volume (optional for measurement only)."
        )

        self.prevVolumeButton = qt.QPushButton("◀")
        self.prevVolumeButton.setFixedWidth(30)
        self.prevVolumeButton.toolTip = "Switch to previous volume"
        self.nextVolumeButton = qt.QPushButton("▶")
        self.nextVolumeButton.setFixedWidth(30)
        self.nextVolumeButton.toolTip = "Switch to next volume"
        volLayout = qt.QHBoxLayout()
        volLayout.addWidget(self.volumeSelector, 1)
        volLayout.addWidget(self.prevVolumeButton)
        volLayout.addWidget(self.nextVolumeButton)
        form.addRow("Volume:", volLayout)

        self.markupSelector = slicer.qMRMLNodeComboBox()
        self.markupSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.markupSelector.selectNodeUponCreation = True
        self.markupSelector.addEnabled = False
        self.markupSelector.removeEnabled = False
        self.markupSelector.noneEnabled = True
        self.markupSelector.showHidden = False
        self.markupSelector.showChildNodeTypes = False
        self.markupSelector.setMRMLScene(slicer.mrmlScene)
        self.markupSelector.setToolTip(
            "Select a Markups Fiducial node to store the landmark points.\n"
            "Use the 'Place' buttons below to create one automatically."
        )
        form.addRow("Markups:", self.markupSelector)

        self.clearMarkupButton = qt.QPushButton("Clear points")
        self.clearMarkupButton.toolTip = "Remove all control points from the selected Markups node."
        form.addRow("", self.clearMarkupButton)

        self.flipXAxisCheckBox = qt.QCheckBox("Correct flip (mirror X-axis)")
        self.flipXAxisCheckBox.toolTip = (
            "Check if the image is displayed mirrored (anterior facing left). "
            "Flips the X coordinate sign in angle computation."
        )
        form.addRow(self.flipXAxisCheckBox)

        # Per-landmark placement grid
        self._landmark_wrapper = qt.QWidget()
        _wl = qt.QVBoxLayout(self._landmark_wrapper)
        _wl.setContentsMargins(0, 0, 0, 0)
        _wl.setSpacing(0)
        self._landmark_inner = None
        self.landmark_rows = {}  # label -> (place_button, status_label)
        form.addRow("Landmarks:", self._landmark_wrapper)
        if initial_set:
            self._rebuild_landmark_grid(initial_set)

        self.showVectorsCheck = qt.QCheckBox("Show vectors (L1, S1, pelvis, L1_pelvis)")
        self.showVectorsCheck.toolTip = (
            "Overlay auxiliary lines for each measurement vector:\n"
            "  L1:       L1_ant → L1_post  (full line)\n"
            "  S1:       S1_ant → S1_post  (full line)\n"
            "  pelvis:   FH → midpoint of S1  (segment)\n"
            "  L1_pelvis: FH → L1_center  (segment)"
        )
        form.addRow(self.showVectorsCheck)

        self.resultsTable = qt.QTableWidget()
        self.resultsTable.setColumnCount(2)
        self.resultsTable.setHorizontalHeaderLabels(["Parameter", "Value (deg)"])
        self.resultsTable.verticalHeader().hide()
        self.resultsTable.horizontalHeader().setStretchLastSection(True)
        if initial_set:
            self._build_results_table(initial_set)
        form.addRow("Results:", self.resultsTable)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        form.addRow(self.statusLabel)

        hintLabel = qt.QLabel("F1: Show hotkeys")
        hintLabel.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow(hintLabel)

    def _rebuild_landmark_grid(self, mset):
        """Rebuild the per-landmark Place button grid for the given measurement set."""
        if self._landmark_inner is not None:
            self._landmark_wrapper.layout().removeWidget(self._landmark_inner)
            self._landmark_inner.setParent(None)
            self._landmark_inner = None

        inner = qt.QWidget()
        grid = qt.QGridLayout(inner)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setSpacing(3)
        grid.setColumnStretch(1, 1)

        self.landmark_rows = {}
        for i, (label, instruction) in enumerate(zip(mset.point_labels, mset.point_instructions)):
            btn = qt.QPushButton("Place")
            btn.setFixedWidth(58)
            btn.setToolTip(f"Enter placement mode for: {label}")

            desc = qt.QLabel(instruction)
            desc.wordWrap = True

            status = qt.QLabel("○")
            status.setFixedWidth(20)
            status.setAlignment(qt.Qt.AlignCenter)
            status.setStyleSheet("color: gray;")

            grid.addWidget(btn, i, 0)
            grid.addWidget(desc, i, 1)
            grid.addWidget(status, i, 2)
            self.landmark_rows[label] = (btn, status)

        self._landmark_wrapper.layout().addWidget(inner)
        self._landmark_inner = inner

    def _build_results_table(self, mset):
        self.resultsTable.setRowCount(len(mset.angle_names))
        for i, name in enumerate(mset.angle_names):
            nameItem = qt.QTableWidgetItem(name)
            nameItem.setFlags(qt.Qt.ItemIsEnabled)
            self.resultsTable.setItem(i, 0, nameItem)
            valueItem = qt.QTableWidgetItem("--")
            valueItem.setFlags(qt.Qt.ItemIsEnabled)
            self.resultsTable.setItem(i, 1, valueItem)

    def update_for_set(self, mset):
        self._rebuild_landmark_grid(mset)
        self._build_results_table(mset)
