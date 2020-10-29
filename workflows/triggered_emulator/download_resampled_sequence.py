#!/usr/bin/env python
# coding: utf-8
import loaders
import numpy as np
import trigger
import xarray as xr

# In[ ]:


url = "gs://vcm-ml-archive/prognostic_runs/2020-09-25-physics-on-free"


# In[2]:


def resample_inactive(x):
    active = trigger.is_active(x).values

    (i_active,) = np.nonzero(active)
    (i_inactive,) = np.nonzero(~active)

    assert len(i_active) < len(i_inactive), len(i_active)

    inactive_sampled = np.random.choice(i_inactive, size=len(i_active), replace=False)
    return xr.concat(
        [x.isel(sample=i_active), x.isel(sample=inactive_sampled)], dim="sample"
    )


# In[3]:


mapper = loaders.mappers.open_baseline_emulator(url)

input_variables = [
    "air_temperature",
    "specific_humidity",
    "cos_zenith_angle",
    "surface_geopotential",
]
output_variables = ["dQ1", "dQ2"]


# In[4]:


sequence = loaders.batches.batches_from_mapper(
    mapper, variable_names=input_variables + output_variables, timesteps_per_batch=5,
)

resampled_seq = sequence.map(resample_inactive)


# In[ ]:


local = resampled_seq.local("output")
