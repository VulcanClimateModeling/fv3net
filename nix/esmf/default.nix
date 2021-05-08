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
  stdenvNoCC
  ,netcdffortran
  ,gfortran
  ,mpich
  ,coreutils
  ,which
  ,llvmPackages
}:
stdenvNoCC.mkDerivation rec {
  pname = "esmf";
  version = "0.0.0";

  src = builtins.fetchGit {
    url = "https://git.code.sf.net/p/esmf/esmf";
    rev = "f5d862d2ec066e76647f53c239b8c58c7af28e45";
  };


  buildPhase = ''
    # set build variables
    export ESMF_DIR=$(pwd)

    export ESMF_INSTALL_PREFIX=$out
    export ESMF_INSTALL_HEADERDIR=$out/include
    export ESMF_INSTALL_MODDIR=$out/include
    export ESMF_INSTALL_LIBDIR=$out/lib
    export ESMF_INSTALL_BINDIR=$out/bin
    export ESMF_INSTALL_DOCDIR=$out/share/docs

    export ESMF_NETCDF_INCLUDE=$netcdffortran/include
    export ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
    export ESMF_BOPT=O3
    export ESMF_CXXCOMPILEOPTS="$ESMF_CXXCOMPILEOPTS -Wno-format-security"

    # compile
    make lib -j8
    make install
    make installcheck
  '';

  # need to fix the linked libraries for some reason.
  # The "id" of these dylibs points to the build directory
  postFixup = stdenvNoCC.lib.optionalString stdenvNoCC.isDarwin  ''
  function fixNameLib {
      install_name_tool -id "$1" "$1"
  }
  fixNameLib $out/lib/libesmf.dylib
  fixNameLib $out/lib/libesmf_fullylinked.dylib 
  '';

  # nativeBuildInputs = [ m4 ];
  # buildInputs = [ hdf5 curl mpi ];
  buildInputs = [ netcdffortran gfortran mpich gfortran.cc coreutils which 
    (lib.optional stdenv.isDarwin llvmPackages.openmp)
  ] ;
  inherit netcdffortran gfortran;
  CXX="${gfortran}/bin/g++";
  CC="${gfortran}/bin/gcc";
  ESMF_CXXCOMPILER="${CXX}";
  ESMF_CCOMPILER="${CC}";



  meta = {
      description = "Libraries for the Unidata network Common Data Format";
      platforms = stdenv.lib.platforms.unix;
      homepage = "https://www.unidata.ucar.edu/software/netcdf/";
      license = {
        url = "https://www.unidata.ucar.edu/software/netcdf/docs/copyright.html";
      };
  };
}
