{
  description = "A very basic flake";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-20.09";
  inputs.utils.url = "github:numtide/flake-utils";
  inputs.poetry2nix-src.url = "github:nix-community/poetry2nix";

  outputs = { self, utils, nixpkgs, poetry2nix-src}: utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; overlays = [ poetry2nix-src.overlay ]; };
    in
    {
      devShell.x86_64-darwin  = pkgs.poetry2nix.mkPoetryPackages {
        projectDir = ./.;
      }.env;
    });
}
