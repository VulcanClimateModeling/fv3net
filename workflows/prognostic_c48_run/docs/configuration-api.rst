.. py:module:: runtime.config
.. _configuration-api:

Configuration API
-----------------

The python model is configured using a nested hierarchy of dataclasses_. This
static structure allows the type-checker to find errors related to improper
access of configurations, and serves as a centralization point for
documentation. :class:`UserConfig` is the top-level configuration object. It
refers to several component configurations.

The model reads these dataclasses from a yaml file format like this:

.. literalinclude:: config-example.yaml
    :language: yaml

These entries are translations to dataclasses. Any yaml entries not-described
in the :class:`UserConfig` are ignored when the model loads its
configurations with :func:`get_config`. See :class:`UserConfig` and the
configuration objects it links to for detailed documentation on the available
configuration options.


Top-level Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: runtime.segmented_run.prepare_config
    :members:

Python run-time configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: runtime.config
    :members: get_config, get_namelist

.. autoclass:: UserConfig

Python "prephysics and postphysics"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:module:: runtime.steppers.machine_learning
.. autoclass:: MachineLearningConfig

.. py:module:: runtime.nudging
.. autoclass:: NudgingConfig

.. py:module:: runtime.steppers.prescriber
.. autoclass:: PrescriberConfig

.. py:module:: runtime.steppers.radiation
.. autoclass:: RadiationStepperConfig


Python Diagnostics
~~~~~~~~~~~~~~~~~~

.. py:module:: runtime.diagnostics.manager
.. autoclass:: DiagnosticFileConfig
.. autoclass:: TimeConfig


Fortran Diagnostics
~~~~~~~~~~~~~~~~~~

.. py:module:: runtime.diagnostics.fortran
.. autoclass:: FortranFileConfig
.. autoclass:: FortranTimeConfig
.. autoclass:: FortranVariableNameSpec

.. _dataclasses: https://docs.python.org/3/library/dataclasses.html