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
            "Select a Markups Fiducial node to store the 5 landmark points.\n"
            "A Markups Fiducial is a named list of 3D points persisted in the Slicer scene.\n"
            "Use 'New / Add point' to create one automatically."
        )
        form.addRow("Markups:", self.markupSelector)

        self.createMarkupButton = qt.QPushButton("New / Add point")
        self.createMarkupButton.toolTip = (
            "Create a Markups Fiducial if needed, then enter single-point placement mode."
        )
        form.addRow("", self.createMarkupButton)

        self.clearMarkupButton = qt.QPushButton("Clear points")
        self.clearMarkupButton.toolTip = "Remove all control points from the selected Markups node."
        form.addRow("", self.clearMarkupButton)

        self.flipXAxisCheckBox = qt.QCheckBox("Correct flip (mirror X-axis)")
        self.flipXAxisCheckBox.toolTip = (
            "Check if the image is displayed mirrored (anterior facing left). "
            "Flips the X coordinate sign in angle computation."
        )
        form.addRow(self.flipXAxisCheckBox)

        self.instructionsLabel = qt.QLabel("")
        self.instructionsLabel.wordWrap = True
        if initial_set:
            self._build_instructions(initial_set)
        form.addRow(self.instructionsLabel)

        self.showVectorsCheck = qt.QCheckBox("Show vectors (L1, S1, pelvis)")
        self.showVectorsCheck.toolTip = (
            "Overlay auxiliary lines for each measurement vector:\n"
            "  L1:     L1_ant → L1_post  (full line)\n"
            "  S1:     S1_ant → S1_post  (full line)\n"
            "  Pelvis: FH → midpoint of S1  (segment)"
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

    def _build_instructions(self, mset):
        lines = "\n".join(f"{i+1}) {desc}" for i, desc in enumerate(mset.point_instructions))
        self.instructionsLabel.setText(f"Place {len(mset.point_labels)} landmarks in order:\n" + lines)

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
        self._build_instructions(mset)
        self._build_results_table(mset)
