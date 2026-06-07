#-----------------------------------------------------------------------------
set(MODULE_NAME CervicalMeasureAssist)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  lib/cervical_assist_controller.py
  lib/logic_angles_cervical.py
  lib/cervical_logic_export.py
  lib/cervical_logic_inference.py
  lib/cervical_measurement_sets.py
  lib/ui_measure.py
  lib/ui_export.py
  lib/ui_auto.py
  lib/__init__.py
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
