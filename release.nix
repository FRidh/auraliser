{ pythonPackages, acoustics, ism, geometry, scintillations, streaming, turbulence }:

with pythonPackages;

buildPythonPackage rec {
  name = "auraliser-${version}";
  version = "0.1dev";

  src = ./.;

  buildInputs = [ cython pytest ];
  propagatedBuildInputs = [ acoustics cytoolz dill geometry ism matplotlib numba numpy scintillations streaming scipy streaming turbulence ];

  doCheck = false;
}
