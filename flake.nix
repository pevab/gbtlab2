{
  description = "GBTLab2 flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = inputs@{ nixpkgs, flake-utils, ...}:
  flake-utils.lib.eachDefaultSystem (
    system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          name = "gbtlab2-dev-shell";
          packages = with pkgs; [
            python312
            python312Packages.uv
          ];
        };
      }
  );
}
