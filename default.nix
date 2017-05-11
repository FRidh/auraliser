{ buildPythonPackage
, pytest
, cython
, acoustics
, cytoolz
, ism
, numpy
, scipy
, matplotlib
, numba
, scintillations
, turbulence
}:

buildPythonPackage rec {
  name = "auraliser-${version}";
  version = "0.0";

  src = ./.;

  preBuild = ''
    make clean
  '';

  checkInputs = [ pytest ];
  buildInputs = [ cython ];

  propagatedBuildInputs = [ acoustics cytoolz ism numpy
    scipy matplotlib numba scintillations turbulence
  ];

  meta = {
    description = "Auraliser";
  };
  doCheck = false;
}

