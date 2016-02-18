{ pythonPackages, acoustics, ism, geometry, scintillations, streaming, turbulence }:

pythonPackages.buildPythonPackage rec {
  name = "auraliser-${version}";
  version = "0.1dev";

  src = ./.;

  buildInputs = with pythonPackages; [ pytest ];
  propagatedBuildInputs = with pythonPackages; [ acoustics cytoolz dill geometry ism matplotlib numba numpy scintillations scipy streaming turbulence ];

  doCheck = false;
}
