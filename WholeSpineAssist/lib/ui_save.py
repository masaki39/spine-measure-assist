"""
保存パネル。手動保存とオートセーブの制御。
"""

from __future__ import annotations

import ctk
import qt


class SaveUI:
    def __init__(self, parent_layout):
        self.button = ctk.ctkCollapsibleButton()
        self.button.text = "Save"
        parent_layout.addWidget(self.button)

        hbox = qt.QHBoxLayout(self.button)
        hbox.setSpacing(6)

        self.saveButton = qt.QPushButton("💾 Save")
        self.saveButton.setToolTip("現在のケースを保存する  [Ctrl+S]")
        self.saveButton.setFixedHeight(28)
        self.saveButton.setStyleSheet("QPushButton { font-weight: bold; }")

        self.autoSaveCheck = qt.QCheckBox("Auto-save on navigate")
        self.autoSaveCheck.setChecked(True)
        self.autoSaveCheck.setToolTip("ケース移動時に自動保存する")

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setStyleSheet("color: #86efac; font-size: 11px;")

        hbox.addWidget(self.saveButton)
        hbox.addWidget(self.autoSaveCheck)
        hbox.addStretch()
        hbox.addWidget(self.statusLabel)

    def set_status(self, msg: str, ok: bool = True) -> None:
        color = "#86efac" if ok else "#fca5a5"
        self.statusLabel.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.statusLabel.setText(msg)

    @property
    def auto_save(self) -> bool:
        return self.autoSaveCheck.isChecked()
