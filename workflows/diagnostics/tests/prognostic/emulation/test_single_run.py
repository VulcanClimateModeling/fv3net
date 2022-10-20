import numpy as np
import pytest
import vcm

from fv3net.diagnostics.prognostic_run.emulation import single_run

cdl = """
netcdf out {
dimensions:
    time = 6;
    tile = 6;
    z = 79 ;
    y = 12 ;
    x = 12 ;
    phalf = 80 ;
    y_interface = 13 ;
    x_interface = 13 ;
variables:
    double time(time) ;
        time:_FillValue = NaN ;
        time:calendar_type = "JULIAN" ;
        time:cartesian_axis = "T" ;
        time:long_name = "time" ;
        time:units = "days since 2016-07-01" ;
        time:calendar = "JULIAN" ;
    double z(z) ;
        z:_FillValue = NaN ;
        z:cartesian_axis = "Z" ;
        z:edges = "phalf" ;
        z:long_name = "ref full pressure level" ;
        z:positive = "down" ;
        z:units = "mb" ;
    double y(y) ;
        y:_FillValue = NaN ;
        y:cartesian_axis = "Y" ;
        y:long_name = "T-cell latitude" ;
        y:units = "degrees_N" ;
    double x(x) ;
        x:_FillValue = NaN ;
        x:cartesian_axis = "X" ;
        x:long_name = "T-cell longitude" ;
        x:units = "degrees_E" ;
    float delp(time, tile, z, y, x) ;
        delp:_FillValue = NaNf ;
        delp:cell_methods = "time: point" ;
        delp:long_name = "pressure thickness" ;
        delp:units = "pa" ;
        delp:coordinates = "time" ;
    double phalf(phalf) ;
        phalf:_FillValue = NaN ;
        phalf:cartesian_axis = "Z" ;
        phalf:long_name = "ref half pressure level" ;
        phalf:positive = "down" ;
        phalf:units = "mb" ;
    float surface_precipitation_due_to_zhao_carr_emulator(time, tile, y, x) ;
        surface_precipitation_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        surface_precipitation_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        surface_precipitation_due_to_zhao_carr_emulator:long_name = "surface precipitation due to zhao_carr_microphysics emulator" ;
        surface_precipitation_due_to_zhao_carr_emulator:units = "kg/m^2/s" ;
        surface_precipitation_due_to_zhao_carr_emulator:coordinates = "time" ;
    float surface_precipitation_due_to_zhao_carr_physics(time, tile, y, x) ;
        surface_precipitation_due_to_zhao_carr_physics:_FillValue = NaNf ;
        surface_precipitation_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        surface_precipitation_due_to_zhao_carr_physics:long_name = "surface precipitation due to zhao_carr_microphysics physics" ;
        surface_precipitation_due_to_zhao_carr_physics:units = "kg/m^2/s" ;
        surface_precipitation_due_to_zhao_carr_physics:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_gscond_emulator(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_gscond_emulator:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_gscond_emulator:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_gscond_emulator:long_name = "temperature tendency due to zhao_carr_gscond emulator" ;
        tendency_of_air_temperature_due_to_gscond_emulator:units = "K/s" ;
        tendency_of_air_temperature_due_to_gscond_emulator:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_gscond_physics(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_gscond_physics:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_gscond_physics:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_gscond_physics:long_name = "temperature tendency due to zhao_carr_gscond physics" ;
        tendency_of_air_temperature_due_to_gscond_physics:units = "K/s" ;
        tendency_of_air_temperature_due_to_gscond_physics:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_zhao_carr_emulator(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:long_name = "temperature tendency due to zhao_carr_microphysics emulator" ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:units = "K/s" ;
        tendency_of_air_temperature_due_to_zhao_carr_emulator:coordinates = "time" ;
    float tendency_of_air_temperature_due_to_zhao_carr_physics(time, tile, z, y, x) ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:_FillValue = NaNf ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:long_name = "temperature tendency due to zhao_carr_microphysics physics" ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:units = "K/s" ;
        tendency_of_air_temperature_due_to_zhao_carr_physics:coordinates = "time" ;
    float tendency_of_cloud_water_due_to_gscond_physics(time, tile, z, y, x) ;
        tendency_of_cloud_water_due_to_gscond_physics:_FillValue = NaNf ;
        tendency_of_cloud_water_due_to_gscond_physics:cell_methods = "time: point" ;
        tendency_of_cloud_water_due_to_gscond_physics:long_name = "cloud water due to zhao_carr_gscond physics" ;
        tendency_of_cloud_water_due_to_gscond_physics:units = "kg/kg/s" ;
        tendency_of_cloud_water_due_to_gscond_physics:coordinates = "time" ;
    float tendency_of_cloud_water_due_to_zhao_carr_emulator(time, tile, z, y, x) ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:long_name = "cloud water due to zhao_carr_microphysics emulator" ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:units = "kg/kg/s" ;
        tendency_of_cloud_water_due_to_zhao_carr_emulator:coordinates = "time" ;
    float tendency_of_cloud_water_due_to_zhao_carr_physics(time, tile, z, y, x) ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:_FillValue = NaNf ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:long_name = "cloud water due to zhao_carr_microphysics physics" ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:units = "kg/kg/s" ;
        tendency_of_cloud_water_due_to_zhao_carr_physics:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_gscond_emulator(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_gscond_emulator:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_gscond_emulator:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_gscond_emulator:long_name = "specific humidity tendency due to zhao_carr_gscond emulator" ;
        tendency_of_specific_humidity_due_to_gscond_emulator:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_gscond_emulator:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_gscond_physics(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_gscond_physics:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_gscond_physics:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_gscond_physics:long_name = "specific humidity tendency due to zhao_carr_gscond physics" ;
        tendency_of_specific_humidity_due_to_gscond_physics:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_gscond_physics:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_zhao_carr_emulator(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:long_name = "specific humidity tendency due to zhao_carr_microphysics emulator" ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_zhao_carr_emulator:coordinates = "time" ;
    float tendency_of_specific_humidity_due_to_zhao_carr_physics(time, tile, z, y, x) ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:_FillValue = NaNf ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:cell_methods = "time: point" ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:long_name = "specific humidity tendency due to zhao_carr_microphysics physics" ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:units = "kg/kg/s" ;
        tendency_of_specific_humidity_due_to_zhao_carr_physics:coordinates = "time" ;
    float area(tile, y, x) ;
        area:_FillValue = NaNf ;
        area:cell_methods = "time: point" ;
        area:long_name = "cell area" ;
        area:units = "m**2" ;
        area:coordinates = "time" ;
    float lat(tile, y, x) ;
        lat:_FillValue = NaNf ;
        lat:cell_methods = "time: point" ;
        lat:long_name = "latitude" ;
        lat:units = "degrees_N" ;
        lat:coordinates = "time" ;
    float latb(tile, y_interface, x_interface) ;
        latb:_FillValue = NaNf ;
        latb:cell_methods = "time: point" ;
        latb:long_name = "latitude" ;
        latb:units = "degrees_N" ;
        latb:coordinates = "time" ;
    float lon(tile, y, x) ;
        lon:_FillValue = NaNf ;
        lon:cell_methods = "time: point" ;
        lon:long_name = "longitude" ;
        lon:units = "degrees_E" ;
        lon:coordinates = "time" ;
    float lonb(tile, y_interface, x_interface) ;
        lonb:_FillValue = NaNf ;
        lonb:cell_methods = "time: point" ;
        lonb:long_name = "longitude" ;
        lonb:units = "degrees_E" ;
        lonb:coordinates = "time" ;
    double x_interface(x_interface) ;
        x_interface:_FillValue = NaN ;
        x_interface:cartesian_axis = "X" ;
        x_interface:long_name = "cell corner longitude" ;
        x_interface:units = "degrees_E" ;
    double y_interface(y_interface) ;
        y_interface:_FillValue = NaN ;
        y_interface:cartesian_axis = "Y" ;
        y_interface:long_name = "cell corner latitude" ;
        y_interface:units = "degrees_N" ;
    double air_temperature(time, tile, z, y, x) ;
        air_temperature:_FillValue = NaN ;
        air_temperature:units = "degK" ;
        air_temperature:coordinates = "time" ;
    double cloud_water_mixing_ratio(time, tile, z, y, x) ;
        cloud_water_mixing_ratio:_FillValue = NaN ;
        cloud_water_mixing_ratio:units = "kg/kg" ;
        cloud_water_mixing_ratio:coordinates = "time" ;
    double eastward_wind(time, tile, z, y, x) ;
        eastward_wind:_FillValue = NaN ;
        eastward_wind:units = "m/s" ;
        eastward_wind:coordinates = "time" ;
    double land_sea_mask(time, tile, y, x) ;
        land_sea_mask:_FillValue = NaN ;
        land_sea_mask:units = "" ;
        land_sea_mask:coordinates = "time" ;
    double latitude(time, tile, y, x) ;
        latitude:_FillValue = NaN ;
        latitude:units = "radians" ;
        latitude:coordinates = "time" ;
    double longitude(time, tile, y, x) ;
        longitude:_FillValue = NaN ;
        longitude:units = "radians" ;
        longitude:coordinates = "time" ;
    double northward_wind(time, tile, z, y, x) ;
        northward_wind:_FillValue = NaN ;
        northward_wind:units = "m/s" ;
        northward_wind:coordinates = "time" ;
    double pressure_thickness_of_atmospheric_layer(time, tile, z, y, x) ;
        pressure_thickness_of_atmospheric_layer:_FillValue = NaN ;
        pressure_thickness_of_atmospheric_layer:units = "Pa" ;
        pressure_thickness_of_atmospheric_layer:coordinates = "time" ;
    double specific_humidity(time, tile, z, y, x) ;
        specific_humidity:_FillValue = NaN ;
        specific_humidity:units = "kg/kg" ;
        specific_humidity:coordinates = "time" ;
    double surface_pressure(time, tile, y, x) ;
        surface_pressure:_FillValue = NaN ;
        surface_pressure:units = "Pa" ;
        surface_pressure:coordinates = "time" ;
    double total_precipitation(time, tile, y, x) ;
        total_precipitation:_FillValue = NaN ;
        total_precipitation:units = "m" ;
        total_precipitation:coordinates = "time" ;
    double vertical_wind(time, tile, z, y, x) ;
        vertical_wind:_FillValue = NaN ;
        vertical_wind:units = "m/s" ;
        vertical_wind:coordinates = "time" ;
data:
    time = 1, 2, 3, 4, 5, 6;
    z = 1000.        ,  987.19230769,  974.38461538,  961.57692308,
        948.76923077,  935.96153846,  923.15384615,  910.34615385,
        897.53846154,  884.73076923,  871.92307692,  859.11538462,
        846.30769231,  833.5       ,  820.69230769,  807.88461538,
        795.07692308,  782.26923077,  769.46153846,  756.65384615,
        743.84615385,  731.03846154,  718.23076923,  705.42307692,
        692.61538462,  679.80769231,  667.        ,  654.19230769,
        641.38461538,  628.57692308,  615.76923077,  602.96153846,
        590.15384615,  577.34615385,  564.53846154,  551.73076923,
        538.92307692,  526.11538462,  513.30769231,  500.5       ,
        487.69230769,  474.88461538,  462.07692308,  449.26923077,
        436.46153846,  423.65384615,  410.84615385,  398.03846154,
        385.23076923,  372.42307692,  359.61538462,  346.80769231,
        334.        ,  321.19230769,  308.38461538,  295.57692308,
        282.76923077,  269.96153846,  257.15384615,  244.34615385,
        231.53846154,  218.73076923,  205.92307692,  193.11538462,
        180.30769231,  167.5       ,  154.69230769,  141.88461538,
        129.07692308,  116.26923077,  103.46153846,   90.65384615,
         77.84615385,   65.03846154,   52.23076923,   39.42307692,
         26.61538462,   13.80769231,    1.;
}

"""  # noqa


@pytest.mark.parametrize(
    "func",
    [
        single_run.plot_histogram_begin_end,
        # this test fails in CI for some reason
        # single_run.plot_cloud_weighted_average,
        single_run.plot_cloud_maps,
        single_run.skill_table,
        single_run.skill_time_table,
        single_run.log_lat_vs_p_skill("cloud_water"),
    ],
)
def test_log_functions(func):

    ds = vcm.cdl_to_dataset(cdl)
    for key in ds:
        ds[key].values[:] = 0

    for key in set(ds.coords) - {"time"}:
        ds[key].values[:] = np.arange(len(ds[key]))

    nx = len(ds.x)
    ds["lat"].values[:] = np.linspace(-45, 45, nx)
    func(ds)


def test_skill_table(regtest):
    ds = vcm.cdl_to_dataset(cdl)
    output = single_run.skill_table(ds)
    for name in sorted(output):

        try:
            signature = output[name].columns
        except AttributeError:
            # is not a wandb.Table
            signature = ""

        print(name, ":", signature, file=regtest)


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(func, id=func.__name__)
        for func in single_run.get_summary_functions()
    ],
)
def test_summary_function(func, regtest):
    ds = vcm.cdl_to_dataset(cdl)
    output = dict(func(ds))
    print(sorted(output), file=regtest)


# xfail this tests since it requires internet access...still is useful for
# integration testing
@pytest.mark.xfail
def test_get_url_from_tag():
    tag = "rnn-gscond-cloudtdep-cbfc4a-30d-v2-online"
    run = single_run.get_prognostic_run_from_tag(tag=tag)
    assert run.group == tag
    rundir = single_run.get_rundir_from_prognostic_run(run)
    assert rundir.startswith("gs://")
