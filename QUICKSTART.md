To start, clean up any lingering build artifacts in the fortran submodule

    cd external/fv3gfs-fortran/ && git clean -fxd

Build image

    VERSION=latest make build_image_prognostic_run 

Prepare run directory (rundir):

    sh prepare_rundir.sh

Enter the docker image:

    docker run -ti --entrypoint bash -v (pwd):/workdir -w /workdir  us.gcr.io/vcm-ml/prognostic_run:latest

This should give you bash prompt in the docker container. fv3.exe is installed to /usr/bin/fv3.exe.

Once inside of the image, change to the rundir, and run the model via the `run.sh` script:

    cd rundir
    sh run.sh

You can check out the logs (saved to `logs` and `err`). The
GFS_physics_driver.F90 currently contains this code demonstrating how to use
the call_py_fort routines:

    call set_state("prsi", Statein%prsi)
    call call_function("builtins", "print")
    call get_state("pris", Statein%prsi)

Call_py_fort basically operates by pushing fortran arrays into a global
python dictionary, calling functions with this dictionary as input, and then
reading numpy arrays from this dictionary back into fortran. Let this
dictionary by called STATE. In terms of python operations, the above lines
roughly translate to

    STATE: Mapping[str, Union[np.ndarray, str, float]]

    # abuse of notation signifyling that the left-hand side is a numpy array
    STATE["prsi"] = Statein%prsi as a numpy array
    # same as `print` but with module name
    builtins.print(STATE)
    # transfer from python back to fortran memory
    Statein%prsi[:] = STATE["prsi"]


Thus, if succesful, the logs should print dictionary with a "prsi" entry for every timestep. Indeed this is what happens:

    {'prsi': array([[9.94275280e+04, 9.98183565e+04, 1.00298827e+05, ...,
        1.00847785e+05, 1.01756696e+05, 1.02275592e+05],
       [9.88979682e+04, 9.92862450e+04, 9.97635772e+04, ...,
        1.00308952e+05, 1.01211927e+05, 1.01727435e+05],
       [9.82987610e+04, 9.86841501e+04, 9.91579304e+04, ...,
        9.96992493e+04, 1.00595508e+05, 1.01107181e+05],
       ...,
       [2.21958000e+02, 2.21958000e+02, 2.21958000e+02, ...,
        2.21958000e+02, 2.21958000e+02, 2.21958000e+02],
       [1.37790000e+02, 1.37790000e+02, 1.37790000e+02, ...,
        1.37790000e+02, 1.37790000e+02, 1.37790000e+02],
       [6.42470000e+01, 6.42470000e+01, 6.42470000e+01, ...,
        6.42470000e+01, 6.42470000e+01, 6.42470000e+01]])}
    
By modifying, the arguments of `call_function` you can call any python
function in the pythonpath.