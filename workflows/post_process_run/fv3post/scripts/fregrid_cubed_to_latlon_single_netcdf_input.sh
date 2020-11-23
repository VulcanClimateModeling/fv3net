#!/bin/bash

input=$1
output=$2

# authenticate with gcloud
if [ -f $GOOGLE_APPLICATION_CREDENTIALS ]
then
    gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
fi

gsutil cp $input /tmp/data.nc
python -m fv3post.scripts.single_netcdf_to_tiled /tmp/data.nc /tmp/tiled_data
fields="$(python -m fv3post.scripts.print_fields /tmp/tiled_data.tile1.nc)"

/usr/bin/regrid.sh /tmp/tiled_data $output C48 $fields --nlon 360 --nlat 180

rm /tmp/data.nc /tmp/tiled_data.tile?.nc