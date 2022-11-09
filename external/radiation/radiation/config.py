from typing import Sequence, Mapping, Hashable, Any, Optional
import dataclasses

LOOKUP_DATA_PATH = "gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz"  # noqa: E501
FORCING_DATA_PATH = "gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/data.tar.gz"  # noqa: 501


@dataclasses.dataclass
class GFSPhysicsControl:
    """A ported version of the Fortran GFS_physics_control structure aka 'model'.
    
    Args:
        levs: Number of model levels.
        levr: Number of model levels.
        ntcw: Tracer index of cloud liquid water.
        ntrw: Tracer index of rain water.
        ntiw: Tracer index of cloud ice water.
        ntsw: Tracer index of snow water.
        ntgl: Tracer index of graupel water.
        ntoz: Tracer index of ozone.
        ntclamt: Tracer index of cloud amount.
        ntrac: Number of tracers.
        nfxr:
        ncld: In physics namelist.
        ncnd: In physics namelist.
        fhswr: Shortwave radiation timestep in seconds. In physics namelist.
        fhlwr: Longwave radiation timestep in seconds. In physics namelist.
        lsswr: Logical flag for SW radiation calculations
        lslwr: Logical flag for LW radiation calculations
        imp_physics: Choice of microphysics scheme:
            11: GFDL microphysics scheme (only ported option)
            8: Thompson microphysics scheme
            10: Morrison-Gettelman microphysics scheme
        lgfdlmprad:
        uni_cld:
        effr_in:
        indcld:
        num_p3d:
        npdf3d:
        ncnvcld3d:
        lmfdeep2:
        lmfshal:
        sup:
        kdt:
        do_sfcperts:
        pertalb:
        do_only_clearsky_rad:
        swhtr: Whether to output SW heating rate. In physics namelist.
        lwhtr: Whether to output LW heating rate. In physics namelist.
        lprnt:
        lssav:
        nsswr: Integer number of physics timesteps between shortwave radiation
            calculations
        nslwr: Integer number of physics timesteps between longwave radiation
            calculations
    """

    levs: Optional[int] = None
    levr: Optional[int] = None
    ntcw: Optional[int] = None
    ntrw: Optional[int] = None
    ntiw: Optional[int] = None
    ntsw: Optional[int] = None
    ntgl: Optional[int] = None
    ntoz: Optional[int] = None
    ntclamt: Optional[int] = None
    ntrac: Optional[int] = None
    nfxr: int = 45
    ncld: int = 5
    ncnd: int = 5
    fhswr: float = 3600.0
    fhlwr: float = 3600.0
    lsswr: bool = True
    lslwr: bool = True
    imp_physics: int = 11
    lgfdlmprad: bool = False
    uni_cld: bool = False
    effr_in: bool = False
    indcld: int = -1
    num_p3d: int = 1
    npdf3d: int = 0
    ncnvcld3d: int = 0
    lmfdeep2: bool = True
    lmfshal: bool = True
    sup: float = 1.0
    kdt: int = 1
    do_sfcperts: bool = False
    pertalb: Sequence[Sequence[float]] = dataclasses.field(
        default_factory=lambda: [[-999.0], [-999.0], [-999.0], [-999.0], [-999.0]]
    )
    do_only_clearsky_rad: bool = False
    swhtr: bool = True
    lwhtr: bool = True
    lprnt: bool = False
    lssav: bool = True
    nsswr: Optional[int] = None
    nslwr: Optional[int] = None

    @classmethod
    def from_physics_namelist(cls, physics_namelist: Mapping[Hashable, Any]):

        PHYSICS_NAMELIST_TO_GFS_CONTROL = {
            "imp_physics": "imp_physics",
            "ncld": "ncld",
            "ncnd": "ncld",
            "fhswr": "fhswr",
            "fhlwr": "fhlwr",
            "swhtr": "swhtr",
            "lwhtr": "lwhtr",
        }

        return cls(
            **_namelist_to_config_args(
                physics_namelist, PHYSICS_NAMELIST_TO_GFS_CONTROL
            )
        )


@dataclasses.dataclass
class RadiationConfig:
    """A configuration class for the radiation wrapper. These namelist flags and
    other attributes control the wrapper behavior. The documentation here is largely
    cut and pasted from the Fortran radiation routines.
    
    Args:
        iemsflg: Surface emissivity control flag. In physics namelist as 'iems'.
        ioznflg: Ozone data source control flag.
        ictmflg: Data IC time/date control flag.
            yyyy#, external data ic time/date control flag
            -2: same as 0, but superimpose seasonal cycle from climatology data set.
            -1: use user provided external data for the forecast time, no
                extrapolation.
            0: use data at initial cond time, if not available, use latest, no
                extrapolation.
            1: use data at the forecast time, if not available, use latest and
                extrapolation.
            yyyy0: use yyyy data for the forecast time no further data extrapolation.
            yyyy1: use yyyy data for the fcst. if needed, do extrapolation to match
                the fcst time.
        isolar: Solar constant cntrl. In physics namelist as 'isol'.
            0: use the old fixed solar constant in "physcon"
            10: use the new fixed solar constant in "physcon"
            1: use noaa ann-mean tsi tbl abs-scale with cycle apprx
            2: use noaa ann-mean tsi tbl tim-scale with cycle apprx
            3: use cmip5 ann-mean tsi tbl tim-scale with cycl apprx
            4: use cmip5 mon-mean tsi tbl tim-scale with cycl apprx
        ico2flg: CO2 data source control flag. In physics namelist as 'ico2'.
        iaerflg: Volcanic aerosols. In physics namelist as 'iaer'.
        ialbflg: Surface albedo control flag. In physics namelist as 'ialb'.
        icldflg:
        ivflip: Vertical index direction control flag for radiation calculations.
            0: Top of model to surface
            1: Surface to top of model
        iovrsw: Cloud overlapping control flag for shortwave radiation. In physics
            namelist as 'iovr_sw'.
            0: random overlapping clouds
            1: maximum/random overlapping clouds
            2: maximum overlap cloud (not implemented in port)
            3: decorrelation-length overlap clouds (not implemented in port)
        iovrlw: Cloud overlapping control flag for longwave radiation. In physics
            namelist as 'iovr_lw'.
            0: random overlapping clouds
            1: maximum/random overlapping clouds
            2: maximum overlap cloud (not implemented in port)
            3: decorrelation-length overlap clouds (not implemented in port)
        isubcsw: Sub-column cloud approx flag in SW radiation. In physics
            namelist as 'isubc_sw'.
            0: no sub-column cloud treatment, use grid-mean cloud quantities
            1: MCICA sub-column, prescribed random numbers
            2: MCICA sub-column, providing optional seed for random numbers
        isubclw: Sub-column cloud approx flag in LW radiation. In physics
            namelist as 'isubc_lw'.
            0: no sub-column cloud treatment, use grid-mean cloud quantities
            1: MCICA sub-column, prescribed random numbers
            2: MCICA sub-column, providing optional seed for random numbers
        lcrick: Control flag for eliminating CRICK.
        lcnorm: Control flag for in-cloud condensate. In namelist as `ccnorm`.
            False: Grid-mean condensate
            True: Normalize grid-mean condensate by cloud fraction
        lnoprec: Precip effect on radiation flag (ferrier microphysics).
        iswcliq: Optical property for liquid clouds for SW.
        gfs_physics_control: GFSPhysicsControl data class
    """

    iemsflg: int = 1
    ioznflg: int = 7
    ictmflg: int = 1
    isolar: int = 2
    ico2flg: int = 2
    iaerflg: int = 111
    ialbflg: int = 1
    icldflg: int = 1
    ivflip: int = 1
    iovrsw: int = 1
    iovrlw: int = 1
    isubcsw: int = 2
    isubclw: int = 2
    lcrick: bool = False
    lcnorm: bool = False
    lnoprec: bool = False
    iswcliq: int = 1
    gfs_physics_control: GFSPhysicsControl = dataclasses.field(
        default_factory=lambda: GFSPhysicsControl()
    )

    @classmethod
    def from_physics_namelist(
        cls, physics_namelist: Mapping[Hashable, Any]
    ) -> "RadiationConfig":
        """Generate RadiationConfig from fv3gfs physics namelist to ensure common keys are
        identical. Remaining values from RadiationConfig defaults.
        """

        gfs_physics_control = GFSPhysicsControl.from_physics_namelist(physics_namelist)

        PHYSICS_NAMELIST_TO_RAD_CONFIG = {
            "iems": "iemsflg",
            "isol": "isolar",
            "ico2": "ico2flg",
            "iaer": "iaerflg",
            "ialb": "ialbflg",
            "iovr_sw": "iovrsw",
            "iovr_lw": "iovrlw",
            "isubc_sw": "isubcsw",
            "isubc_lw": "isubclw",
            "ccnorm": "lcnorm",
        }

        return cls(
            **dict(
                **_namelist_to_config_args(
                    physics_namelist, PHYSICS_NAMELIST_TO_RAD_CONFIG
                ),
                gfs_physics_control=gfs_physics_control
            )
        )


def _namelist_to_config_args(
    namelist: Mapping[Hashable, Any], arg_mapping: Mapping[str, str]
) -> Mapping[str, Any]:
    config_args = {}
    for namelist_entry, config_arg in arg_mapping.items():
        if namelist_entry in namelist:
            config_args[config_arg] = namelist[namelist_entry]
    return config_args
