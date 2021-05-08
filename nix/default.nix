# Copied from https://nixos.org/nixos/nix-pills/callpackage-design-pattern.html
let
  # nixpkgs = import <nixpkgs> {};
  # allPkgs = nixpkgs // pkgs;
  # callPackage = path: overrides:
  #   let f = import path;
  #   in f ((builtins.intersectAttrs (builtins.functionArgs f) allPkgs) // overrides);
  packageOverrides = mpi: fv3: self: super: rec {
    wrapper = super.callPackage ./wrapper.nix {inherit fv3;};
    mpi4py = super.mpi4py.override {inherit mpi;};
    cftime = super.cftime.overrideAttrs (oldAttrs: {
        pname = "cftime";
        version="1.2.1";
        src = super.fetchPypi {
          pname = "cftime";
          version="1.2.1";
          sha256 = "ab5d5076f7d3e699758a244ada7c66da96bae36e22b9e351ce0ececc36f0a57f";
        };
      });
    numcodecs = super.numcodecs.overrideAttrs (oldAttrs: {
      disabledTests = [
        "test_backwards_compatibility"
        "test_encode_decode"
        "test_legacy_codec_broken"
        "test_bytes"
      ];
    });
    fv3gfs-util = super.buildPythonPackage {
      version = "0.1.0";
      pname = "fv3gfs-wrapper";
      src = ../external/fv3gfs-util;
      propagatedBuildInputs = [
        super.fsspec
        super.xarray
        super.zarr
        super.typing-extensions
        cftime
      ];

    };
    fv3config = super.buildPythonPackage rec {
      pname = "fv3config";
      version = "0.6.1";

      src = super.fetchPypi {
        inherit pname version;
        sha256 = "00f0c1c1lnbhwmyg1kjvp0yxbj1kj3pls7jh2zywb0l0fs72n2px";
      };

      doCheck = false;

      propagatedBuildInputs = [
        super.fsspec
        super.backoff
        super.pyyaml
        super.appdirs
        # gcsfs
        # f90nml
        # dacite
      ];
    };
    # gcsfs = super.callPackage ../pynixify/packages/gcsfs {} ;
    # dacite = super.callPackage ../pynixify/packages/dacite {} ;
    # f90nml = super.callPackage ../pynixify/packages/f90nml {} ;
  };
  overlay = self: super: with nixpkgs; rec {
    # ensure that everything uses mpich for mpi
    fms = super.callPackage ./fms { };
    esmf = super.callPackage ./esmf { };
    nceplibs = super.callPackage ./nceplibs { };
    fv3 = super.callPackage ./fv3 { };
    python3 = super.python3.override {
      packageOverrides = packageOverrides mpich fv3;
    };
    wrapper = python.pkgs.wrapper;
  };
  nixpkgs = import (builtins.fetchTarball {
  # Descriptive name to make the store path easier to identify
  name = "release-20.09";
  # Commit hash for nixos-unstable as of 2018-09-12
  url = "https://github.com/nixos/nixpkgs/archive/20.09.tar.gz";
  # Hash obtained using `nix-prefetch-url --unpack <url>`
  sha256 = "1wg61h4gndm3vcprdcg7rc4s1v3jkm5xd7lw8r2f67w502y94gcy";
}) {overlays = [overlay];};
  shell = with nixpkgs; mkShell {
  buildInputs = [
          pkg-config
          fv3
          fms
          esmf
          nceplibs
          netcdf
          netcdffortran
          lapack
          blas
          gfortran 
          # see this issue: https://discourse.nixos.org/t/building-shared-libraries-with-fortran/11876/2
          gfortran.cc.lib
          llvmPackages.openmp
          mpich
          # debugging
          gdb

          # for cartopy
          geos 
          proj_5
          poetry
    ];

    MPI="mpich";
    FC="gfortran";
    inherit fms;
    CC="${gfortran}/bin/gcc";
    inherit gfortran;
    gfc="${gfortran.cc}";
    cclib="${gfortran.cc.lib}";

  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.

    export PIP_CONSTRAINT=$(pwd)/constraints.txt
    export PIP_NO_BINARY=shapely,cartopy,mpi4py
    export GEOS_CONFIG=geos-config

    export LD_LIBRARY_PATH="$cclib/lib"
    # need this for shapely to work
    export DYLD_LIBRARY_PATH=${geos}/lib
    export CC=${gfortran.cc}/bin/gcc
    # source .env/bin/activate
  '';
};
in shell
