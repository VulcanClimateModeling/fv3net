stdenv: package:
stdenv.mkDerivation {
  src = package.src;
  name = "${package.pname}-${package.version}.tar.gz";
  buildInputs = [ package.pythonModule.pkgs.setuptools ];
  builder = ./builder.sh;
}

