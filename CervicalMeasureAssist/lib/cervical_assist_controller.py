import logging
import math
import os

import numpy as np
import qt
import slicer
import vtk

from logic_angles_cervical import REQUIRED_KEYS
from cervical_logic_export import ExportLogic
from cervical_logic_inference import OnnxInferenceLogic
from cervical_measurement_sets import CERVICAL_SET, get_set, set_names


def _extend_to_line(p1, p2, extend=1000.0):
    """Extend a segment outward in both directions to approximate an infinite line."""
    dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-6:
        return p1, p2
    nx, ny, nz = dx/length, dy/length, dz/length
    q1 = (p1[0]-nx*extend, p1[1]-ny*extend, p1[2]-nz*extend)
    q2 = (p2[0]+nx*extend, p2[1]+ny*extend, p2[2]+nz*extend)
    return q1, q2


class AssistController:
    """
    Glue code between MeasureUI / ExportUI and the underlying logic.
    Keeps the entrypoint thin and encapsulates event handling.
    """

    def __init__(self, measure_ui, export_ui, auto_ui, logic):
        self.measure_ui = measure_ui
        self.export_ui = export_ui
        self.auto_ui = auto_ui
        self.logic = logic
        self._active_set = CERVICAL_SET
        self.counter = 1
        self.infer = OnnxInferenceLogic()
        self._observedMarkupNode = None
        self._markupObserverTags = []
        self._heatmap_channels = None
        self._heatmap_volume_node = None
        self._vector_line_nodes = {}
        self._pending_label = None
        self._shortcuts = []
        self._connect_signals()
        self._setup_shortcuts()
        self._update_counter_preview()

    # --- Shortcuts ---
    def _setup_shortcuts(self):
        mw = slicer.util.mainWindow()
        bindings = [
            (",", self.onVolumePrev),
            (".", self.onVolumeNext),
            ("r", self.onRunInference),
            ("e", self.onExportCsv),
            ("F1", self.onShowHotkeys),
        ]
        for key, slot in bindings:
            sc = qt.QShortcut(qt.QKeySequence(key), mw)
            sc.connect("activated()", slot)
            self._shortcuts.append(sc)

        space_sc = qt.QShortcut(qt.QKeySequence(qt.Qt.Key_Space), mw)
        space_sc.setContext(qt.Qt.ApplicationShortcut)
        space_sc.connect("activated()", self._on_place_next)
        self._shortcuts.append(space_sc)

    # --- Signal wiring ---
    def _connect_signals(self):
        if self.measure_ui.setCombo is not None:
            self.measure_ui.setCombo.connect("currentIndexChanged(int)", self.onSetChanged)
        self.measure_ui.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeChanged)
        self.measure_ui.prevVolumeButton.connect("clicked()", self.onVolumePrev)
        self.measure_ui.nextVolumeButton.connect("clicked()", self.onVolumeNext)
        self.measure_ui.clearMarkupButton.connect("clicked()", self.onClearMarkups)
        self.measure_ui.markupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMarkupNodeChanged)
        self.measure_ui.flipXAxisCheckBox.connect("toggled(bool)", lambda *_: self.onUpdateMeasurements())
        self.measure_ui.showVectorsCheck.connect("toggled(bool)", self._onShowVectorsToggled)

        self.export_ui.csvButton.connect("clicked()", self.onExportCsv)
        self.export_ui.browseButton.connect("clicked()", self.onBrowse)
        self.export_ui.prefixEdit.textChanged.connect(lambda *_: self._update_counter_preview())
        self.export_ui.caseIdSourceCombo.connect(
            "currentIndexChanged(int)", self._onCaseIdSourceChanged
        )
        self.auto_ui.modelBrowseButton.connect("clicked()", self.onBrowseModel)
        self.auto_ui.runButton.connect("clicked()", self.onRunInference)
        self.auto_ui.heatmapCheckBox.connect("toggled(bool)", self.onHeatmapToggled)
        self.auto_ui.landmarkCombo.connect("currentIndexChanged(int)", self.onHeatmapLandmarkChanged)
        self.auto_ui.opacitySlider.connect("valueChanged(int)", self.onHeatmapOpacityChanged)

        self._connect_landmark_buttons()

    def _connect_landmark_buttons(self):
        for label, (btn, _) in self.measure_ui.landmark_rows.items():
            btn.connect("clicked()", lambda lbl=label: self.onPlaceLandmark(lbl))

    # --- Volume navigation ---
    def _volume_nodes(self):
        return [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                if not n.GetHideFromEditors()]

    def onVolumePrev(self):
        nodes = self._volume_nodes()
        if not nodes:
            return
        current = self.measure_ui.volumeSelector.currentNode()
        idx = nodes.index(current) if current in nodes else 0
        self.measure_ui.volumeSelector.setCurrentNode(nodes[(idx - 1) % len(nodes)])

    def onVolumeNext(self):
        nodes = self._volume_nodes()
        if not nodes:
            return
        current = self.measure_ui.volumeSelector.currentNode()
        idx = nodes.index(current) if current in nodes else -1
        self.measure_ui.volumeSelector.setCurrentNode(nodes[(idx + 1) % len(nodes)])

    # --- Handlers ---
    def onSetChanged(self, index):
        name = self.measure_ui.setCombo.itemText(index)
        self._active_set = get_set(name)
        for vnode in self._vector_line_nodes.values():
            vnode.SetDisplayVisibility(0)
        self._vector_line_nodes = {}
        self.measure_ui.update_for_set(self._active_set)
        self._connect_landmark_buttons()
        self.auto_ui.update_landmark_combo(self._active_set.point_labels)
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is not None:
            self._assignLandmarkLabels(markupNode)
        self.onUpdateMeasurements()

    def onVolumeChanged(self, volumeNode):
        slicer.util.setSliceViewerLayers(background=volumeNode)
        if volumeNode is None:
            return
        self._fill_case_id_from_source(volumeNode)
        if self.auto_ui.autoRunCheck.isChecked() and self.auto_ui.modelPathEdit.text.strip():
            self.onRunInference()

    def _get_patient_id(self, volumeNode):
        try:
            instance_uids = volumeNode.GetAttribute("DICOM.instanceUIDs")
            if not instance_uids:
                return None
            uid = instance_uids.split()[0]
            return slicer.dicomDatabase.instanceValue(uid, "0010,0020").strip() or None
        except Exception:
            return None

    def _get_filename(self, volumeNode):
        try:
            instance_uids = volumeNode.GetAttribute("DICOM.instanceUIDs")
            if not instance_uids:
                return None
            uid = instance_uids.split()[0]
            filepath = slicer.dicomDatabase.fileForInstance(uid)
            if not filepath:
                return None
            return os.path.splitext(os.path.basename(filepath))[0]
        except Exception:
            return None

    def _fill_case_id_from_source(self, volumeNode):
        if volumeNode is None:
            return
        idx = self.export_ui.caseIdSourceCombo.currentIndex
        if idx == 1:
            value = self._get_filename(volumeNode)
        elif idx == 0:
            value = self._get_patient_id(volumeNode)
        else:
            value = None
        self.export_ui.caseIdEdit.setText(value if value else "")

    def _onCaseIdSourceChanged(self, index):
        volumeNode = self.measure_ui.volumeSelector.currentNode()
        self._fill_case_id_from_source(volumeNode)

    def onPlaceLandmark(self, label):
        self._pending_label = label
        fiducialNode = self._ensureMarkupNodeExists()
        self.measure_ui.markupSelector.setCurrentNode(fiducialNode)

        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(fiducialNode.GetID())

        slicer.modules.markups.logic().StartPlaceMode(0)
        self.measure_ui.statusLabel.setText(f"Placement mode: click to place  {label}")
        self._highlight_pending_button(label)

    def _highlight_pending_button(self, active_label):
        for label, (btn, _) in self.measure_ui.landmark_rows.items():
            if label == active_label:
                btn.setText("Placing…")
                btn.setStyleSheet("background-color: #d06000; color: white;")
            else:
                btn.setText("Place")
                btn.setStyleSheet("")

    def _reset_pending_button(self):
        for btn, _ in self.measure_ui.landmark_rows.values():
            btn.setText("Place")
            btn.setStyleSheet("")

    def onShowHotkeys(self):
        qt.QMessageBox.information(
            slicer.util.mainWindow(), "Hotkeys",
            "Space      次の未設定ランドマークを配置\n"
            ",          Previous volume\n"
            ".          Next volume\n"
            "r          Run inference\n"
            "e          Export CSV\n"
            "F1         Show this help",
        )

    def _on_place_next(self):
        """Space: 次の未設定ランドマークに配置モードを切り替える。"""
        markupNode = self.measure_ui.markupSelector.currentNode()
        placed = set()
        if markupNode is not None:
            placed = {markupNode.GetNthControlPointLabel(i)
                      for i in range(markupNode.GetNumberOfControlPoints())}
        for label in self._active_set.point_labels:
            if label not in placed:
                self.onPlaceLandmark(label)
                return
        self.measure_ui.statusLabel.setText("全ランドマーク設定済みです")

    def _update_landmark_status_buttons(self, markupNode):
        placed = set()
        if markupNode is not None:
            for i in range(markupNode.GetNumberOfControlPoints()):
                placed.add(markupNode.GetNthControlPointLabel(i))
        for label, (_, status) in self.measure_ui.landmark_rows.items():
            if label in placed:
                status.setText("✓")
                status.setStyleSheet("color: #00aa00; font-weight: bold;")
            else:
                status.setText("○")
                status.setStyleSheet("color: gray;")

    def onClearMarkups(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            self.measure_ui.statusLabel.setText("No Markups node selected.")
            return
        markupNode.RemoveAllControlPoints()
        self.measure_ui.statusLabel.setText("All points cleared.")

    def onMarkupNodeChanged(self, node):
        if self._observedMarkupNode is not None:
            for tag in self._markupObserverTags:
                self._observedMarkupNode.RemoveObserver(tag)
        self._markupObserverTags = []
        self._observedMarkupNode = node

        for vnode in self._vector_line_nodes.values():
            vnode.SetDisplayVisibility(0)

        if node is not None:
            self._ensureVectorNodesExist()
            tag = node.AddObserver(slicer.vtkMRMLMarkupsNode.PointAddedEvent, self._onPointAdded)
            self._markupObserverTags.append(tag)
            for event in [
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
            ]:
                tag = node.AddObserver(event, lambda *_: self.onUpdateMeasurements())
                self._markupObserverTags.append(tag)
            self._assignLandmarkLabels(node)

        self.onUpdateMeasurements()

    def _onPointAdded(self, *_):
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            return
        n = markupNode.GetNumberOfControlPoints()
        if n == 0:
            return

        if self._pending_label is not None:
            label = self._pending_label
            self._pending_label = None
            self._reset_pending_button()
            for i in range(n - 1):
                if markupNode.GetNthControlPointLabel(i) == label:
                    markupNode.RemoveNthControlPoint(i)
                    break
            markupNode.SetNthControlPointLabel(
                markupNode.GetNumberOfControlPoints() - 1, label
            )
        else:
            used = {markupNode.GetNthControlPointLabel(i) for i in range(n - 1)}
            for label in self._active_set.point_labels:
                if label not in used:
                    markupNode.SetNthControlPointLabel(n - 1, label)
                    break

        self.onUpdateMeasurements()

    def onUpdateMeasurements(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        self._update_landmark_status_buttons(markupNode)

        if markupNode is None:
            return

        label_to_pos = {}
        label_to_ras3d = {}
        coordsRAS = [0.0, 0.0, 0.0]
        active_labels = set(self._active_set.point_labels)
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            if label in active_labels:
                markupNode.GetNthControlPointPosition(i, coordsRAS)
                x = -coordsRAS[0] if self.measure_ui.flipXAxisCheckBox.isChecked() else coordsRAS[0]
                label_to_pos[label] = (x, coordsRAS[1])
                label_to_ras3d[label] = (coordsRAS[0], coordsRAS[1], coordsRAS[2])

        n_placed = len(label_to_pos)
        n_total = len(self._active_set.point_labels)

        missing_required = [k for k in REQUIRED_KEYS if k not in label_to_pos]
        if missing_required:
            self.measure_ui.statusLabel.setText(f"Landmarks: {n_placed} / {n_total}")
            for vnode in self._vector_line_nodes.values():
                vnode.SetDisplayVisibility(0)
            return

        try:
            angles = self.logic.compute_angles_from_points(label_to_pos)
        except ValueError as exc:
            self.measure_ui.statusLabel.setText(f"Error: {exc}")
            return

        self._updateResultsTable(angles)
        status_text = "Measurements updated."
        if n_placed < n_total:
            status_text += f"  ({n_placed} / {n_total} points placed)"
        self.measure_ui.statusLabel.setText(status_text)
        self._updateVectorOverlays(label_to_ras3d)

    def onBrowseModel(self):
        file_path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), "Select ONNX model", "", "ONNX (*.onnx)"
        )
        if file_path:
            self.auto_ui.modelPathEdit.setText(file_path)

    def onRunInference(self):
        volumeNode = self.measure_ui.volumeSelector.currentNode()
        if volumeNode is None:
            self.auto_ui.statusLabel.setText("Error: no volume selected.")
            return

        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            markupNode = self._ensureMarkupNodeExists()
            self.measure_ui.markupSelector.setCurrentNode(markupNode)

        model_path = self.auto_ui.modelPathEdit.text.strip()
        if not model_path:
            self.auto_ui.statusLabel.setText("Error: no ONNX model path specified.")
            return

        try:
            self.infer.load_model(model_path)
            coords_ij, heatmap_2d = self.infer.predict_and_place(volumeNode, markupNode)
        except Exception as exc:
            logging.exception("Inference failed")
            self.auto_ui.statusLabel.setText(f"Error: inference failed ({exc})")
            return

        self._heatmap_channels = heatmap_2d
        self._show_heatmap_overlay(volumeNode, heatmap_2d)
        for w in [self.auto_ui.heatmapCheckBox, self.auto_ui.landmarkCombo, self.auto_ui.opacitySlider]:
            w.setEnabled(True)
        self.auto_ui.statusLabel.setText("Inference complete. Landmarks placed in Markups.")
        self.onUpdateMeasurements()

    def onBrowse(self):
        directory = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "Select output folder"
        )
        if directory:
            self.export_ui.outputDirEdit.setText(directory)

    def onExport(self):
        volumeNode = self.measure_ui.volumeSelector.currentNode()
        markupNode = self.measure_ui.markupSelector.currentNode()
        outputDir = self.export_ui.outputDirEdit.text.strip()
        manualCaseId = self.export_ui.caseIdEdit.text.strip()

        if volumeNode is None:
            self.export_ui.exportStatusLabel.setText("Error: no volume selected.")
            return
        if markupNode is None:
            self.export_ui.exportStatusLabel.setText("Error: no Markups node selected.")
            return
        if not outputDir:
            self.export_ui.exportStatusLabel.setText("Error: no output folder specified.")
            return

        caseId = manualCaseId if manualCaseId else self._find_next_case_id(outputDir)
        if caseId is None:
            self.export_ui.exportStatusLabel.setText("Error: no available case ID found.")
            return

        try:
            exporter = ExportLogic(flip_x_axis=self.measure_ui.flipXAxisCheckBox.isChecked(), measurement_set=self._active_set)
            result = exporter.export_training_sample(
                volumeNode=volumeNode,
                markupNode=markupNode,
                outputDir=outputDir,
                caseId=caseId,
                overwrite=self.export_ui.overwriteCheck.isChecked(),
            )
        except ValueError as exc:
            self.export_ui.exportStatusLabel.setText(f"Error: {exc}")
            return
        except Exception:
            logging.exception("Export failed")
            self.export_ui.exportStatusLabel.setText("Error: export failed. See Python Console for details.")
            return

        self.export_ui.exportStatusLabel.setText(
            "Export complete: "
            f"{os.path.basename(result['npy'])}, "
            f"{os.path.basename(result['json'])}"
        )
        if not manualCaseId:
            self.counter += 1
            self._update_counter_preview()

    def onExportCsv(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        outputDir = self.export_ui.outputDirEdit.text.strip()
        manualCaseId = self.export_ui.caseIdEdit.text.strip()

        if markupNode is None:
            self.export_ui.exportStatusLabel.setText("Error: no Markups node selected.")
            return
        if not outputDir:
            self.export_ui.exportStatusLabel.setText("Error: no output folder specified.")
            return

        caseId = manualCaseId if manualCaseId else self._format_counter_preview()
        overwrite = self.export_ui.overwriteCheck.isChecked()

        try:
            exporter = ExportLogic(flip_x_axis=self.measure_ui.flipXAxisCheckBox.isChecked(), measurement_set=self._active_set)
            csv_path = exporter.export_angles_csv(markupNode=markupNode, outputDir=outputDir, caseId=caseId, overwrite=overwrite)
        except ValueError as exc:
            self.export_ui.exportStatusLabel.setText(f"Error: {exc}")
            return
        except Exception:
            logging.exception("CSV export failed")
            self.export_ui.exportStatusLabel.setText("Error: CSV export failed. See Python Console for details.")
            return

        action = "updated" if overwrite else "appended"
        status = f"CSV {action}: {os.path.basename(csv_path)}"

        if self.export_ui.exportTrainingDataCheck.isChecked():
            volumeNode = self.measure_ui.volumeSelector.currentNode()
            if volumeNode is None:
                self.export_ui.exportStatusLabel.setText(f"{status} | Error: no volume for training data.")
                return
            try:
                result = exporter.export_training_sample(
                    volumeNode=volumeNode,
                    markupNode=markupNode,
                    outputDir=outputDir,
                    caseId=caseId,
                    overwrite=overwrite,
                )
                status += f" + {os.path.basename(result['npy'])}"
            except Exception:
                logging.exception("Training data export failed")
                self.export_ui.exportStatusLabel.setText(f"{status} | Error: training export failed. See Python Console.")
                return

        self.export_ui.exportStatusLabel.setText(status)

    # --- Vector overlay ---
    def _updateVectorOverlays(self, label_to_ras3d):
        pts = dict(label_to_ras3d)
        for key, (a, b) in self._active_set.midpoint_definitions.items():
            if a in label_to_ras3d and b in label_to_ras3d:
                pa, pb = label_to_ras3d[a], label_to_ras3d[b]
                pts[key] = ((pa[0]+pb[0])/2, (pa[1]+pb[1])/2, (pa[2]+pb[2])/2)

        visible = self.measure_ui.showVectorsCheck.isChecked()

        for name, (p1_key, p2_key) in self._active_set.vector_definitions.items():
            if p1_key not in pts or p2_key not in pts:
                if name in self._vector_line_nodes:
                    self._vector_line_nodes[name].SetDisplayVisibility(0)
                continue

            node = self._vector_line_nodes.get(name)
            if node is None:
                continue
            ep1 = pts[p1_key]
            ep2 = pts[p2_key]
            if self._active_set.vector_modes[name] == "Line":
                ep1, ep2 = _extend_to_line(ep1, ep2)

            n_pts = node.GetNumberOfControlPoints()
            if n_pts < 2:
                wasModifying = node.StartModify()
                node.RemoveAllControlPoints()
                node.AddControlPoint(list(ep1))
                node.AddControlPoint(list(ep2))
                node.EndModify(wasModifying)
            else:
                node.SetNthControlPointPosition(0, ep1[0], ep1[1], ep1[2])
                node.SetNthControlPointPosition(1, ep2[0], ep2[1], ep2[2])

            node.SetDisplayVisibility(int(visible))

    def _onShowVectorsToggled(self, checked):
        for vnode in self._vector_line_nodes.values():
            vnode.SetDisplayVisibility(int(checked))

    # --- Heatmap helpers ---
    def _for_each_slice(self, callback):
        lm = slicer.app.layoutManager()
        for name in lm.sliceViewNames():
            callback(lm.sliceWidget(name).sliceLogic().GetSliceCompositeNode())

    def onHeatmapToggled(self, checked):
        vol_id = self._heatmap_volume_node.GetID() if checked and self._heatmap_volume_node is not None else ""
        self._for_each_slice(lambda cn: cn.SetForegroundVolumeID(vol_id))

    def onHeatmapLandmarkChanged(self, index):
        if self._heatmap_channels is None or self._heatmap_volume_node is None:
            return
        if index == 0:
            hm_2d = np.max(self._heatmap_channels, axis=0)
        else:
            hm_2d = self._heatmap_channels[index - 1]
        slicer.util.updateVolumeFromArray(self._heatmap_volume_node, hm_2d[np.newaxis])

    def onHeatmapOpacityChanged(self, value):
        self.auto_ui.opacityLabel.setText(f"{value}%")
        self._for_each_slice(lambda cn: cn.SetForegroundOpacity(value / 100.0))

    def _show_heatmap_overlay(self, volumeNode, heatmap_channels):
        idx = self.auto_ui.landmarkCombo.currentIndex
        hm_2d = np.max(heatmap_channels, axis=0) if idx == 0 else heatmap_channels[idx - 1]
        hm_volume = hm_2d[np.newaxis]

        hm_node = slicer.mrmlScene.GetFirstNodeByName("HeatmapOverlay")
        if hm_node is None:
            hm_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "HeatmapOverlay")
            hm_node.SetHideFromEditors(True)
        self._heatmap_volume_node = hm_node

        slicer.util.updateVolumeFromArray(hm_node, hm_volume)

        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        hm_node.SetIJKToRASMatrix(mat)

        if hm_node.GetDisplayNode() is None:
            hm_node.CreateDefaultDisplayNodes()
        displayNode = hm_node.GetDisplayNode()
        color_node = slicer.mrmlScene.GetFirstNodeByName("ColdToHot")
        if color_node:
            displayNode.SetAndObserveColorNodeID(color_node.GetID())
        else:
            displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindowLevelMinMax(0.0, 1.0)

        if self.auto_ui.heatmapCheckBox.isChecked():
            opacity = self.auto_ui.opacitySlider.value / 100.0
            hm_id = hm_node.GetID()
            def _apply(cn):
                cn.SetForegroundVolumeID(hm_id)
                cn.SetForegroundOpacity(opacity)
            self._for_each_slice(_apply)

    # --- Private helpers ---
    def _ensureVectorNodesExist(self):
        for name in self._active_set.vector_definitions:
            if name in self._vector_line_nodes:
                continue
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", f"Vec_{name}")
            node.SetLocked(True)
            node.SetHideFromEditors(True)
            node.CreateDefaultDisplayNodes()
            node.SetDisplayVisibility(0)
            dn = node.GetDisplayNode()
            if dn:
                r, g, b = self._active_set.vector_colors[name]
                dn.SetSelectedColor(r, g, b)
                dn.SetColor(r, g, b)
                dn.SetTextScale(0)
            self._vector_line_nodes[name] = node

    def _ensureMarkupNodeExists(self):
        current = self.measure_ui.markupSelector.currentNode()
        if current and current.IsA("vtkMRMLMarkupsFiducialNode"):
            return current
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        displayNode = fiducialNode.GetDisplayNode()
        if displayNode:
            displayNode.SetSelectedColor(1.0, 0.4, 0.0)
            displayNode.SetGlyphScale(1.5)
        return fiducialNode

    def _assignLandmarkLabels(self, markupNode):
        labels = self._active_set.point_labels
        used = set()
        unlabeled = []
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            if label in labels:
                used.add(label)
            else:
                unlabeled.append(i)
        available = [lbl for lbl in labels if lbl not in used]
        for idx, label in zip(unlabeled, available):
            markupNode.SetNthControlPointLabel(idx, label)

    def _updateResultsTable(self, anglesDict):
        for i, name in enumerate(self._active_set.angle_names):
            value = anglesDict.get(name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                text = "--"
            else:
                unit = self._active_set.value_units.get(name, "°")
                text = f"{value:.1f}{unit}"
            self.measure_ui.resultsTable.item(i, 1).setText(text)

    def _format_counter_preview(self):
        prefix = self.export_ui.prefixEdit.text.strip() or "case"
        return f"{prefix}{self.counter:03d}"

    def _update_counter_preview(self):
        self.export_ui.set_next_id_preview(self._format_counter_preview())

    def _find_next_case_id(self, outputDir):
        prefix = self.export_ui.prefixEdit.text.strip() or "case"
        for idx in range(self.counter, 10000):
            candidate = f"{prefix}{idx:03d}"
            npy = os.path.join(outputDir, f"{candidate}_image.npy")
            json_path = os.path.join(outputDir, f"{candidate}_landmarks.json")
            nrrd_path = os.path.join(outputDir, f"{candidate}_volume.nrrd")
            if self.export_ui.overwriteCheck.isChecked():
                return candidate
            if not (os.path.exists(npy) or os.path.exists(json_path) or os.path.exists(nrrd_path)):
                return candidate
        return None

    def cleanup(self):
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts = []
        for node in self._vector_line_nodes.values():
            slicer.mrmlScene.RemoveNode(node)
        self._vector_line_nodes = {}
