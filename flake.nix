{
  description = "QIFy - A Python QIF library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pythonPkgs = pkgs.python310Packages;
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.pyright
          pythonPkgs.numpy
          pythonPkgs.pandas
          pythonPkgs.multimethod
        ];
      };

      # packages.${system}.default = with pythonPkgs; {
      #   qify = buildPythonPackage rec {
      #     pname = "qify";
      #     version = "0.0.1";
      #     src = ./.;
      #     
      #     nativeBuildInputs = [ setuptools ];

      #     meta = {
      #       description = "QIFy - A Python QIF library";
      #       license = "";
      #     };

      #     # Placeholder for the sha256 hash, replace with the actual hash
      #     sha256 = "...";
      #   };
      # };
    };
}
