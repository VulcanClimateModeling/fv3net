.. _config usage:

Configuration Usage
-------------------

The prognostic run can can be configured to run with the following

#. :ref:`Baseline (no ML or nudging) <baseline>`
#. :ref:`Nudge-to-fine <nudge to fine>`
#. :ref:`Nudge-to-obs <nudge to obs>`
#. :ref:`Machine learning (prognostic) <ml config>`

The prognostic run provides a command line script ``prepare_config.py`` to
minimize the boilerplate required to configure a run. This script allows
specifying changes over the "default" configurations stored `here <https://github.com/VulcanClimateModeling/fv3net/tree/master/external/fv3kube/fv3kube/base_yamls>`_.

.. _baseline:

Baseline Run
~~~~~~~~~~~~

To configure a simple baseline run, save the following to a file ``minimal.yaml``

.. literalinclude:: prognostic_config.yml
    :language: yaml

This file contains a subset of options described by fv3config_. This file can
contain both fv3config_ settings like ``namelist``, as well as the python
runtime configurations described in :ref:`configuration-api`. To generate a
"full configuration" usable by fv3config_ and the python runtime
configurations, run the following::

    python3 prepare_config.py \
        minimal.yaml \
        gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
        20160801.001500 \
        > fv3config.yaml


The output file ``fv3config.yaml`` (which is to long to include in these
docs) is now compatible with ``fv3config.write_run_directory`` or
:ref:`execution`.

.. _nudge to fine:

Nudge to fine
~~~~~~~~~~~~~

A nudged-to-fine run can be configured by setting the
:py:attr:`runtime.config.UserConfig.nudging` configuration option. This can
be done by adding the following section to the ``minimal.yaml`` file::

    nudging:
        restarts_path: gs://vcm-ml-experiments/2020-06-02-fine-res/coarsen_restarts
        timescale_hours:
            air_temperature: 3
            specific_humidity: 3
            x_wind: 3
            y_wind: 3
            pressure_thickness_of_atmospheric_layer: 3

Notice how these configurations correspond with
:py:attr:`runtime.config.UserConfig.nudging`. Refer to those docs as a
reference.

In addition to the above settings, to prescribe the sea surface temperatures to
the values in the reference restart files, the
``gfs_physics_nml.use_climatological_sst`` namelist parameter must be set to
``false``::

    namelist:
        gfs_physics_nml:
            use_climatological_sst: false

.. _nudge to obs:

Nudge to obs
~~~~~~~~~~~~

Obervational nudging is implemented in the Fortran model. It is activated by
setting the namelist parameter ``fv_core_nml.nudge`` to ``True``. The nudging
is configured through the ``fv_nwp_nudge_nml`` namelist. For convenience, a
base YAML (``v0.6``) is provided which provides useful defaults nudge to obs
runs. The ``gfs_analysis_data`` section defines the location and naming of the
reference analysis data. Here is an example ``minimal.yaml``:

.. literalinclude:: nudge_to_obs_config.yml
    :language: yaml

.. note::

    Nudge-to-obs is not mutually exclusive with any of the first three
    options as it is conducted within the Fortran physics routine.

.. _ml config:

Machine learning
~~~~~~~~~~~~~~~~

A machine learning run can be configured in two ways. The first is by
specifying a path to a fv3fit_ model in
:py:attr:`runtime.config.UserConfig.scikit_learn.model`. This can be done
by adding the following to the ``minimal.yaml`` example::

    scikit_learn:
        model: ["path/to/model"]


For convenient scripting, the ``--model_url`` command line argument adds a
model to :py:class:`runtime.steppers.machine_learning.MachineLearningConfig`.
It can be used multiple times to specify multiple models. For example::

    python3 prepare_config.py \
        minimal.yaml \
        gs://vcm-ml-code-testing-data/c48-restarts-for-e2e \
        20160801.001500 \
        --model_url path/to/model
        --model_url path/to_another/model
        > fv3config.yaml
 
Diagnostics
~~~~~~~~~~~

Python diagnostics
^^^^^^^^^^^^^^^^^^

If no :py:attr:`UserConfig.diagnostics` section is provided in the ``minimal.yaml``,
default diagnostics
are configured depending on whether ML, nudge-to-fine, nudge-to-obs, or baseline runs
are chosen. To save custom diagnostics, provide a ``diagnostics`` section. To save 
additional
tendencies and storages across physics and nudging/ML time steps, add
:py:attr:`UserConfig.step_tendency_variables` and
:py:attr:`UserConfig.step_storage_variables` entries to specify these
variables. Then add an additional output .zarr which includes among its
variables the desired tendencies and/or path storages of these variables due
to physics (``_due_to_fv3_physics``) and/or ML/nudging (``_due_to_python``).

Note that the diagnostic output named ``state_after_timestep.zarr`` is a special case;
it can only be used to save variables that have getters in the wrapper.

This example configures a run with stepwise tendency outputs for several
variables. These tendencies are averaged online over 3 hour intervals before
being saved.

.. code-block:: yaml

    step_tendency_variables: 
        - air_temperature
        - specific_humidity
        - eastward_wind
        - northward_wind
        - cloud_water_mixing_ratio
    step_storage_variables: 
        - specific_humidity
        - cloud_water_mixing_ratio
    diagnostics:
    - name: step_diags.zarr
      chunks:
        time: 4
      times:
        kind: interval-average
        frequency: 10800  # 3 hours = 10800 seconds
      variables:
        - tendency_of_cloud_water_mixing_ratio_due_to_fv3_physics
        - storage_of_specific_humidity_path_due_to_fv3_physics
        - storage_of_cloud_water_mixing_ratio_path_due_to_fv3_physics
        - storage_of_specific_humidity_path_due_to_python


Fortran diagnostics
^^^^^^^^^^^^^^^^^^^

Diagnostics to be output by the Fortran model are specified in the
:py:attr:`UserConfig.fortran_diagnostics` section. This section will be converted
to the Fortran ``diag_table`` representation of diagnostics (see fv3config_ docs).


.. _fv3config: https://fv3config.readthedocs.io/en/latest/
.. _fv3fit: https://vulcanclimatemodeling.com/docs/fv3fit/
