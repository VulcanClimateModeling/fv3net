## Workflow for computing quantities needed to infer fine resolution budgets

This dataflow pipeline computes the quantities needed for computing estimates
of Q1 and Q2.

For Q1 these are the re-coarse-grained:

- C384 air temperature
- C384 omega
- Product of the C384 air temperature and omega
- C384 vertical eddy flux of temperature
- C384 tendency of temperature due to all parameterized physics
- C384 tendency of temperature due to saturation adjustment within the
  dynamical core

For Q2 these are the re-coarse-grained:

- C384 specific humidity
- C384 omega
- Product of the C384 specific humidity and omega
- C384 vertical eddy flux of specific humidity
- C384 tendency of specific humidity due to all parameterized physics
- C384 tendency of specific humidity due to saturation adjustment within the
  dynamical core

The vertical eddy flux convergence terms at the target resolution, and
subsequently Q1 and Q2, are then computed at data access time within the
`FineResolutionSources` mapper.  The default target resolution in the
re-coarse-graining process is C48.

### Local usage

To test locally use this:

    python -m budget \
        gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
        gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
        gs://vcm-ml-scratch/noah/2020-05-12/

Parallel usage:

	 python -m budget \
          gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr/ \
          gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr \
          gs://vcm-ml-scratch/noah/2020-05-18 \
          --runner Direct --direct_num_workers 8 --direct_running_mode multi_processing

This produces the following outputs:
```
$ gsutil ls gs://vcm-ml-scratch/noah/2020-05-12/                                                                                                                                                                                                                       (base)
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile1.nc
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile2.nc
gs://vcm-ml-scratch/noah/2020-05-12/20160801.000730.tile3.nc
gs://vcm-ml-scratch/noah/2020-05-12//
```


### Remote dataflow usage

To coarsen the 40 day ShiELd run data execute the following command in this
directory:

    ./submit.sh

This was last run on Dec 7.  This workflow is run infrequently so often requires
some maintenance (ask whoever last touched this folder in git).