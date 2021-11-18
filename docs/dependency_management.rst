.. _dependency_management:

Dependency management
=====================

There have been a lot of issues related to loose package requirements
and relatively young packages.  To reduce the impacts of frequent
upstream breaking changes, strategies to lock dependencies were put into
place.  To add or update dependencies, see below!

Adding or updating dependencies
-------------------------------

The package ``conda-lock`` is used to ensure deterministic builds anaconda
builds. Therefore, adding or modifying a dependency involves a few steps:

#. add any anaconda packages to the ``environment.yml``
#. add any pip packages to ``pip-requirements.txt``
#. add any pip packages for a ``external/<package>`` to ``external/<package>.requirements.in``
#. run ``make lock_deps`` to create lock files ``conda-<system>.lock`` 
   which explicitly list all the conda packages
#. Commit the lock files and any other changes to git

..  note::
    
    Not all ``setup.py`` files are compatible with `pip-compile`. Packages
    within fv3net can be fixed, but this is not possible or easy for external
    dependencies or submodules. To pin the transitive dependencies of external
    submodules (e.g. external/fv3gfs-util), you can add the requirements to
    ``external/<package>.requirements.in``.

The ``make create_environment`` uses these lock files and
``pip-requirements.txt`` to install its dependencies. It does NOT directly
install the ``environment.yml`` file since that can lead to non-deterministic
builds, and difficult to debug errors in CI.

Pip
^^^

Pip dependencies are specified in a variety of places. Mostly ``setup.py``
files and ``requirements.txt`` files for the dockerfiles. A package called
``pip-tools`` is used to ensure that these files do not conflict with one
another. If they do not, the ``make lock_deps`` rule will generate a file
``constraints.txt`` containing a list of versions to use for pip packages.
These constraints should then be used whenenver ``pip`` is invoked like this::

    pip install -c constraints.txt <other pip args>

Where to pin dependencies?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Since ``constraints.txt`` is compiled automatically, it should not be manually
edited. If you need to constrain or pin a dependency, you should do so in the
``requirements.txt`` used by the build process for the container where the
problem occurs or in the root level ``pip-requirements.txt`` file. 

For instance, suppose ``fsspec`` v0.7.0 breaks some aspect of the prognostic
run image, then you could add something like the following to the
``docker/prognostic_run/requirements.txt``::

    fsspec!=0.7.0

Then run ``make lock_deps`` to update the ``constraints.txt`` file.

This currently requires ``pip`` version < 20.3. The latest version 20.3 does not work with 
the automatically generated ``constraints.txt`` because it contains extras.