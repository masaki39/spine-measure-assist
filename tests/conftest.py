import os
import sys
from unittest.mock import MagicMock

for mod in ["slicer", "vtk", "qt", "ctk"]:
    sys.modules.setdefault(mod, MagicMock())

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "slicer", "LumbarMeasureAssist", "lib"))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

cervical_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "slicer", "CervicalMeasureAssist", "lib"))
if cervical_lib_path not in sys.path:
    sys.path.insert(0, cervical_lib_path)
