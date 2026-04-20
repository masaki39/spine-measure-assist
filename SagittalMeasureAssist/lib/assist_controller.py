import logging
import math
import os

import qt
import slicer
import vtk

from logic_export import ExportLogic, REQUIRED_LABELS_ORDERED
from logic_inference import OnnxInferenceLogic

_VECTOR_COLORS = {
    "L1": (0.2, 0.8, 1.0),     # cyan
    "S1": (1.0, 0.6, 0.1),     # orange
    "pelvis": (0.4, 1.0, 0.4), # green
}
# L1/S1: full line (extended); pelvis: segment only
_VECTOR_MODES = {"L1": "Line", "S1": "Line", "pelvis": "Segment"}


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
        self.counter = 1
        self.infer = OnnxInferenceLogic()
        self._observedMarkupNode = None
        self._markupObserverTags = []
        self._heatmap_channels = None  # (L, H, W)
        self._heatmap_volume_node = None
        self._heatmap_volume_node_ref = None
        self._vector_line_nodes = {}  # key: "L1" / "S1" / "pelvis"
        self._shortcuts = []
        self._connect_signals()
        self._setup_shortcuts()
        self._update_counter_preview()

    # --- Shortcuts ---
    def _setup_shortcuts(self):
        bindings = [
            (",", self.onVolumePrev),
            (".", self.onVolumeNext),
            ("r", self.onRunInference),
            ("e", self.onExport),
        ]
        mw = slicer.util.mainWindow()
        for key, slot in bindings:
            sc = qt.QShortcut(qt.QKeySequence(key), mw)
            sc.connect("activated()", slot)
            self._shortcuts.append(sc)

    # --- Signal wiring ---
    def _connect_signals(self):
        self.measure_ui.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeChanged)
        self.measure_ui.prevVolumeButton.connect("clicked()", self.onVolumePrev)
        self.measure_ui.nextVolumeButton.connect("clicked()", self.onVolumeNext)
        self.measure_ui.createMarkupButton.connect("clicked()", self.onCreateMarkup)
        self.measure_ui.clearMarkupButton.connect("clicked()", self.onClearMarkups)
        self.measure_ui.markupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMarkupNodeChanged)
        self.measure_ui.flipXAxisCheckBox.connect("toggled(bool)", lambda *_: self.onUpdateMeasurements())
        self.measure_ui.showVectorsCheck.connect("toggled(bool)", self._onShowVectorsToggled)

        self.export_ui.exportButton.connect("clicked()", self.onExport)
        self.export_ui.browseButton.connect("clicked()", self.onBrowse)
        self.export_ui.prefixEdit.textChanged.connect(lambda *_: self._update_counter_preview())
        self.auto_ui.modelBrowseButton.connect("clicked()", self.onBrowseModel)
        self.auto_ui.runButton.connect("clicked()", self.onRunInference)
        self.auto_ui.heatmapCheckBox.connect("toggled(bool)", self.onHeatmapToggled)
        self.auto_ui.landmarkCombo.connect("currentIndexChanged(int)", self.onHeatmapLandmarkChanged)
        self.auto_ui.opacitySlider.connect("valueChanged(int)", self.onHeatmapOpacityChanged)

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
    def onVolumeChanged(self, volumeNode):
        slicer.util.setSliceViewerLayers(background=volumeNode)
        if volumeNode is None:
            return
        patient_id = self._get_patient_id(volumeNode)
        if patient_id:
            self.export_ui.caseIdEdit.setText(patient_id)

    def _get_patient_id(self, volumeNode):
        try:
            instance_uids = volumeNode.GetAttribute("DICOM.instanceUIDs")
            if not instance_uids:
                return None
            uid = instance_uids.split()[0]
            return slicer.dicomDatabase.instanceValue(uid, "0010,0020").strip() or None
        except Exception:
            return None

    def onCreateMarkup(self):
        fiducialNode = self._ensureMarkupNodeExists()
        self.measure_ui.markupSelector.setCurrentNode(fiducialNode)
        slicer.modules.markups.logic().StartPlaceMode(0)
        self.measure_ui.statusLabel.text = "Placement mode: click to place one point. Press the button again to add the next point."

    def onClearMarkups(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            self.measure_ui.statusLabel.text = "No Markups node selected."
            return
        markupNode.RemoveAllControlPoints()
        self.measure_ui.statusLabel.text = "All points cleared."

    def onMarkupNodeChanged(self, node):
        if self._observedMarkupNode is not None:
            for tag in self._markupObserverTags:
                self._observedMarkupNode.RemoveObserver(tag)
        self._markupObserverTags = []
        self._observedMarkupNode = node

        for vnode in self._vector_line_nodes.values():
            vnode.SetDisplayVisibility(0)

        if node is not None:
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
        used = {markupNode.GetNthControlPointLabel(i) for i in range(n - 1)}
        for label in REQUIRED_LABELS_ORDERED:
            if label not in used:
                markupNode.SetNthControlPointLabel(n - 1, label)
                break
        self.onUpdateMeasurements()

    def onUpdateMeasurements(self):
        markupNode = self.measure_ui.markupSelector.currentNode()
        if markupNode is None:
            return

        label_to_pos = {}    # 2D (x, y) for angle computation, with optional x-flip
        label_to_ras3d = {}  # 3D RAS for vector visualization
        coordsRAS = [0.0, 0.0, 0.0]
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            if label in REQUIRED_LABELS_ORDERED:
                markupNode.GetNthControlPointPosition(i, coordsRAS)
                x = -coordsRAS[0] if self.measure_ui.flipXAxisCheckBox.isChecked() else coordsRAS[0]
                label_to_pos[label] = (x, coordsRAS[1])
                label_to_ras3d[label] = (coordsRAS[0], coordsRAS[1], coordsRAS[2])

        if len(label_to_pos) < len(REQUIRED_LABELS_ORDERED):
            self.measure_ui.statusLabel.text = f"Landmarks: {len(label_to_pos)} / {len(REQUIRED_LABELS_ORDERED)}"
            for vnode in self._vector_line_nodes.values():
                vnode.SetDisplayVisibility(0)
            return

        points = {label: label_to_pos[label] for label in REQUIRED_LABELS_ORDERED}

        try:
            angles = self.logic.compute_angles_from_points(points)
        except ValueError as exc:
            self.measure_ui.statusLabel.text = f"Error: {exc}"
            return

        self._updateResultsTable(angles)
        self.measure_ui.statusLabel.text = "Measurements updated."
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

        target_h = int(self.auto_ui.heightSpin.value)
        target_w = int(self.auto_ui.widthSpin.value)

        try:
            self.infer.load_model(model_path, (target_h, target_w))
            coords_ij, heatmap_2d = self.infer.predict_and_place(volumeNode, markupNode)
        except Exception as exc:
            logging.exception("Inference failed")
            self.auto_ui.statusLabel.setText(f"Error: inference failed ({exc})")
            return

        self._heatmap_channels = heatmap_2d  # (L, H, W)
        self._heatmap_volume_node_ref = volumeNode
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
            self.export_ui.outputDirEdit.text = directory

    def onExport(self):
        volumeNode = self.measure_ui.volumeSelector.currentNode()
        markupNode = self.measure_ui.markupSelector.currentNode()
        outputDir = self.export_ui.outputDirEdit.text.strip()
        manualCaseId = self.export_ui.caseIdEdit.text.strip()

        if volumeNode is None:
            self.export_ui.exportStatusLabel.text = "Error: no volume selected."
            return
        if markupNode is None:
            self.export_ui.exportStatusLabel.text = "Error: no Markups node selected."
            return
        if not outputDir:
            self.export_ui.exportStatusLabel.text = "Error: no output folder specified."
            return

        caseId = manualCaseId if manualCaseId else self._find_next_case_id(outputDir)
        if caseId is None:
            self.export_ui.exportStatusLabel.text = "Error: no available case ID found."
            return

        try:
            exporter = ExportLogic(flip_x_axis=self.measure_ui.flipXAxisCheckBox.isChecked())
            result = exporter.export_training_sample(
                volumeNode=volumeNode,
                markupNode=markupNode,
                outputDir=outputDir,
                caseId=caseId,
                overwrite=self.export_ui.overwriteCheck.isChecked(),
            )
        except ValueError as exc:
            self.export_ui.exportStatusLabel.text = f"Error: {exc}"
            return
        except Exception:
            logging.exception("Export failed")
            self.export_ui.exportStatusLabel.text = "Error: export failed. See Python Console for details."
            return

        self.export_ui.exportStatusLabel.text = (
            "Export complete: "
            f"{os.path.basename(result['npy'])}, "
            f"{os.path.basename(result['json'])}"
        )
        if not manualCaseId:
            self.counter += 1
            self._update_counter_preview()

    # --- Vector overlay ---
    def _updateVectorOverlays(self, label_to_ras3d):
        s1_ant = label_to_ras3d["S1_ant"]
        s1_post = label_to_ras3d["S1_post"]
        s1_mid = (
            (s1_ant[0] + s1_post[0]) / 2,
            (s1_ant[1] + s1_post[1]) / 2,
            (s1_ant[2] + s1_post[2]) / 2,
        )
        pts = dict(label_to_ras3d, _S1_mid=s1_mid)

        definitions = {
            "L1":     ("L1_ant", "L1_post"),
            "S1":     ("S1_ant", "S1_post"),
            "pelvis": ("FH", "_S1_mid"),
        }
        visible = self.measure_ui.showVectorsCheck.isChecked()

        for name, (p1_key, p2_key) in definitions.items():
            if name not in self._vector_line_nodes:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", f"Vec_{name}")
                node.SetLocked(True)
                node.CreateDefaultDisplayNodes()
                dn = node.GetDisplayNode()
                if dn:
                    r, g, b = _VECTOR_COLORS[name]
                    dn.SetSelectedColor(r, g, b)
                    dn.SetColor(r, g, b)
                    dn.SetTextScale(0)
                self._vector_line_nodes[name] = node

            node = self._vector_line_nodes[name]
            ep1 = pts[p1_key]
            ep2 = pts[p2_key]
            if _VECTOR_MODES[name] == "Line":
                ep1, ep2 = _extend_to_line(ep1, ep2)

            n_pts = node.GetNumberOfControlPoints()
            if n_pts < 2:
                node.RemoveAllControlPoints()
                node.AddControlPoint(list(ep1))
                node.AddControlPoint(list(ep2))
            else:
                node.SetNthControlPointPosition(0, ep1[0], ep1[1], ep1[2])
                node.SetNthControlPointPosition(1, ep2[0], ep2[1], ep2[2])

            node.SetDisplayVisibility(1 if visible else 0)

    def _onShowVectorsToggled(self, checked):
        for vnode in self._vector_line_nodes.values():
            vnode.SetDisplayVisibility(1 if checked else 0)

    # --- Heatmap helpers ---
    def onHeatmapToggled(self, checked):
        layoutManager = slicer.app.layoutManager()
        for sliceName in layoutManager.sliceViewNames():
            compositeNode = layoutManager.sliceWidget(sliceName).sliceLogic().GetSliceCompositeNode()
            if checked and self._heatmap_volume_node is not None:
                compositeNode.SetForegroundVolumeID(self._heatmap_volume_node.GetID())
            else:
                compositeNode.SetForegroundVolumeID("")

    def onHeatmapLandmarkChanged(self, index):
        if self._heatmap_channels is None or self._heatmap_volume_node is None:
            return
        import numpy as np
        if index == 0:
            hm_2d = np.max(self._heatmap_channels, axis=0)
        else:
            hm_2d = self._heatmap_channels[index - 1]
        slicer.util.updateVolumeFromArray(self._heatmap_volume_node, hm_2d[np.newaxis])

    def onHeatmapOpacityChanged(self, value):
        self.auto_ui.opacityLabel.setText(f"{value}%")
        layoutManager = slicer.app.layoutManager()
        for sliceName in layoutManager.sliceViewNames():
            compositeNode = layoutManager.sliceWidget(sliceName).sliceLogic().GetSliceCompositeNode()
            compositeNode.SetForegroundOpacity(value / 100.0)

    def _show_heatmap_overlay(self, volumeNode, heatmap_channels):
        import numpy as np
        idx = self.auto_ui.landmarkCombo.currentIndex
        hm_2d = np.max(heatmap_channels, axis=0) if idx == 0 else heatmap_channels[idx - 1]
        hm_volume = hm_2d[np.newaxis]  # (1, H, W)

        hm_node = slicer.mrmlScene.GetFirstNodeByName("HeatmapOverlay")
        if hm_node is None:
            hm_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "HeatmapOverlay")
            hm_node.SetHideFromEditors(True)  # exclude from volume selectors
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

        opacity = self.auto_ui.opacitySlider.value / 100.0
        layoutManager = slicer.app.layoutManager()
        for sliceName in layoutManager.sliceViewNames():
            compositeNode = layoutManager.sliceWidget(sliceName).sliceLogic().GetSliceCompositeNode()
            compositeNode.SetForegroundVolumeID(hm_node.GetID())
            compositeNode.SetForegroundOpacity(opacity)

    # --- Private helpers ---
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
        used = set()
        unlabeled = []
        for i in range(markupNode.GetNumberOfControlPoints()):
            label = markupNode.GetNthControlPointLabel(i)
            if label in REQUIRED_LABELS_ORDERED:
                used.add(label)
            else:
                unlabeled.append(i)
        available = [l for l in REQUIRED_LABELS_ORDERED if l not in used]
        for idx, label in zip(unlabeled, available):
            markupNode.SetNthControlPointLabel(idx, label)

    def _updateResultsTable(self, anglesDict):
        params = ["PI", "PT", "SS", "LL"]
        for i, name in enumerate(params):
            value = anglesDict.get(name, float("nan"))
            text = "--" if math.isnan(value) else f"{value:.1f}°"
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
