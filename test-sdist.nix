with import <nixpkgs> { };
let f = import ./nix/buildSdist;
in f stdenv python3Packages.requests
