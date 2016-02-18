with import <nixpkgs> {};

let
  python = "python35";
  pythonPackages = pkgs.${python+"Packages"};
  mypkgs = import ~/Code/configuration/nix-config/packages { pythonPackages=pythonPackages; };

  dependencies =  with pythonPackages; with mypkgs; [ acoustics cytoolz dill geometry ism matplotlib numba numpy scintillations scipy streaming turbulence pytest];

in pythonPackages.buildPythonPackage {
  name = "auraliser";
  src = ./.;

  propagatedBuildInputs = dependencies ++ mypkgs.base;
}
