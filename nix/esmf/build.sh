source $stdenv/setup


# copy from read-only git repo
mkdir -p $out
cp -r $src src
chmod -R +w src
cd src

# set build variables
export ESMF_DIR=$(pwd)
export ESMF_INSTALL_PREFIX=$out
export ESMF_NETCDF_INCLUDE=$netcdffortran/include
export ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
export ESMF_BOPT=O3
export ESMF_CXXCOMPILEOPTS="$ESMF_CXXCOMPILEOPTS -Wno-format-security"

# compile
make lib -j8
make install
make installcheck
