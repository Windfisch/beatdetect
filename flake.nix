{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    jack_client.url = "github:spatialaudio/jackclient-python/0.5.5";
    jack_client.flake = false;
  };

  outputs =
    inputs:
    let
      system = "x86_64-linux";
      pkgs = import inputs.nixpkgs { inherit system; };

      jack_client_package = pkgs.python3Packages.buildPythonPackage rec {
        name = "jack_client";

        src = inputs.jack_client;

        propagatedBuildInputs = with pkgs.python3Packages; [
          cffi
        ];

        pyproject = true;
        build-system = [ pkgs.python3Packages.setuptools ];

      };

      beatdetect-lib = pkgs.python3Packages.buildPythonPackage rec {
        name = "beatdetect";

        src = inputs.self;

        propagatedBuildInputs = with pkgs.python3Packages; [
          scipy
          numpy
          matplotlib
          soundfile
          jack_client_package
          wxpython
        ];

        pyproject = true;
        build-system = [ pkgs.python3Packages.setuptools ];

      };

      beatdetect-exe = pkgs.writeShellApplication {
        name = "beatdetect";
        text = ''
          export PIPEWIRE_QUANTUM='4096/48000'
          ${
            (pkgs.python3.withPackages (ps: [
              beatdetect-lib
            ]))
          }/bin/python -m beatdetect "$@"
        '';
      };
    in
    {
      packages.${system} = {
        beatdetect-lib = beatdetect-lib;
        default = beatdetect-exe;
        beatdetect = beatdetect-exe;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python3
          beatdetect-lib
        ];
      };
    };

}
