import ctk
import qt


class ExportUI:
    """
    Builds the export panel (output dir, IDs, overwrite, trigger button).
    """

    def __init__(self, parentLayout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Export (Training Data)"
        parentLayout.addWidget(self.button)

        form = qt.QFormLayout(self.button)

        self.outputDirEdit = qt.QLineEdit()
        self.outputDirEdit.placeholderText = "Output folder (e.g. /path/to/dataset)"
        self.browseButton = qt.QPushButton("Browse...")
        dirLayout = qt.QHBoxLayout()
        dirLayout.addWidget(self.outputDirEdit, 1)
        dirLayout.addWidget(self.browseButton)
        form.addRow("Output dir:", dirLayout)

        self.caseIdEdit = qt.QLineEdit()
        self.caseIdEdit.placeholderText = "Manual case ID (e.g. K16) — leave blank for auto"
        self.caseIdEdit.setToolTip(
            "Manual case ID. Overrides auto-numbering.\n"
            "Auto-filled from DICOM Patient ID when a volume is loaded."
        )
        form.addRow("Case ID:", self.caseIdEdit)

        self.prefixEdit = qt.QLineEdit("case")
        self.prefixEdit.setToolTip(
            "Prefix for auto-generated IDs (e.g. 'case' → case001, case002...).\n"
            "The counter increments after each successful export."
        )
        self.overwriteCheck = qt.QCheckBox("Overwrite existing")
        self.overwriteCheck.checked = False
        self.nextIdLabel = qt.QLabel("case001")
        autoLayout = qt.QHBoxLayout()
        autoLayout.addWidget(qt.QLabel("Prefix:"))
        autoLayout.addWidget(self.prefixEdit)
        autoLayout.addWidget(qt.QLabel("Next ID:"))
        autoLayout.addWidget(self.nextIdLabel)
        form.addRow("Auto-numbering:", autoLayout)
        form.addRow("", self.overwriteCheck)

        self.exportButton = qt.QPushButton("Export training data")
        self.exportButton.toolTip = "Export .npy / .nrrd and landmark JSON with computed angles."

        self.csvButton = qt.QPushButton("Append to CSV")
        self.csvButton.toolTip = (
            "Append the current case's angles to angles.csv in the output directory.\n"
            "Creates the file if it does not exist; adds a header row on first write."
        )

        btnLayout = qt.QHBoxLayout()
        btnLayout.addWidget(self.exportButton)
        btnLayout.addWidget(self.csvButton)
        form.addRow(btnLayout)

        self.exportStatusLabel = qt.QLabel("")
        self.exportStatusLabel.wordWrap = True
        form.addRow(self.exportStatusLabel)

    def set_next_id_preview(self, text):
        self.nextIdLabel.setText(text)
