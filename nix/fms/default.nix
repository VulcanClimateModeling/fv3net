with import <nixpkgs> {};
# { stdenv
# , netcdffortran
# , perl
# , gfortran
# , lapack
# , blas
# , openmpi
# , fetchgit
# }:
# 
{
  stdenv
  ,bash
  ,rsync
  ,gfortran
  ,mpich
  ,automake
  ,autoconf
  ,m4
  ,libtool
  ,bats
  ,netcdffortran
  ,netcdf
} :
stdenv.mkDerivation rec {
  pname = "fms";
  version = "0.0.0";

  src = ../../external/fv3gfs-fortran/FMS/.;

  # nativeBuildInputs = [ m4 ];
  # buildInputs = [ hdf5 curl mpi ];
  buildInputs = [ bash rsync gfortran mpich automake autoconf m4 libtool bats netcdffortran netcdf 
    (lib.optional stdenv.isDarwin llvmPackages.openmp)
  ];
  inherit netcdffortran;

  configurePhase = ''
    mkdir m4
    autoreconf --install

    export CC=mpicc
    export FC=mpifort
    export LOG_DRIVER_FLAGS="--comments"
    export CPPFLAGS="-Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS"
    export FCFLAGS="-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp -I$netcdffortran/include"

    ./configure --prefix=$out
  '';

  meta = {
      description = "Libraries for the Unidata network Common Data Format";
      platforms = stdenv.lib.platforms.unix;
      homepage = "https://www.unidata.ucar.edu/software/netcdf/";
      license = {
        url = "https://www.unidata.ucar.edu/software/netcdf/docs/copyright.html";
      };
  };
}
