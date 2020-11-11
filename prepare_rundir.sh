write_run_directory external/fv3gfs-fortran/tests/pytest/config/default.yml rundir

# write a runscript for convenience
cat << EOF > rundir/run.sh
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/local/esmf/lib/libO3/Linux.gfortran.64.mpiuni.default/:/FMS/libFMS/.libs/
mpirun -np 6 --allow-run-as-root --oversubscribe --mca btl_vader_single_copy_mechanism none fv3.exe > logs 2> err
EOF
