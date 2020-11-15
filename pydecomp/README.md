Pydecomp Overview
=================

## Presentation
**Pydecomp** is a set of python3 scripts that allow the user to decompose
multidimensional data into relevant formats. The technical background of the
methods proposed in this software can be found in Lucas Lestandi PhD. thesis :
"Low rank approximation techniques and Reduced Order Modeling applied to some
fluid dynamics problems". All numerical experiments involving decomposition in
this thesis have been performed using this software. The synthetic data examples
can be replicated easily relying on `2D_benchmark.py` for bivariate experiments
and `benchmark_multivariable.py` for multivariate problems.

The **spirit** of this project is to provide a free and versatile tool to the
scientific community for data decomposition. It is by no means an optimized
version and comes with no warranty. The documentation will be improved through
time but the code has been documented while being developed which should be
sufficient for users already familiar with python and data decomposition
techniques.

## Installation
 - clone the present repository
 - either link manually or install by running `setup.py`

## Acknowledgements
This program was developed by Diego Britez and Lucas Lestandi at I2M Bordeaux,
Laboratoire TREFLE. The authors would like to thank the many members of the lab
that contributed indirectly to the development of this program. In particular
Mejdi Azaiez for his mathematical guidance.


## Code organization
Here, we present, the current organization of the software. It is separated in the following softwares.
 - `/`: 		setup, licence, and test shortcut
 - `/core`: the basic features of pydecomp, easy to export, almost standalone.
	 - the required data formats including **canonical**, **tucker**, **TT**, **QTT**, **Hierarchical** 	and **recursive**
	 - the decomposition methods including **PGD**, **HOSVD**, **TTSVD**, **RPOD**
		, **HT-SVD**, **QTT-SVD**, **SVD** and **POD**
 - `/utils`: many routines that are actually very important for running the code including basic inouts, integration rules, synthetic data generation
 - `/interfaces`: interfaces with common data format for importing and exporting
	of scientific data (matlab, vtk, adios bp)
 - `/analysis`: prewritten benchmarks together with plotting and tests on output
	data
 - `/interpolation`: a simple interpolation ROM for those interested.
 - `/deprecated`: deprecated files that should disappear.

## Documentation
The code is mostly documented internally, with relavant name and
many docstrings. The initial attempt at sphinx documentation was abandonned. Relevant details can be found in the below publications

## More information on the science
[1] L. Lestandi, *“Low rank approximation techniques and reduced order modeling applied to some fluid dynamics problems”* Université de Bordeaux, 2018.

[2] M. Azaïez, L. Lestandi, and T. C. Rebollo, *“Low Rank Approximation of Multidimensional Data In: Pirozzoli S., Sengupta T. (eds) High-Performance Computing of Big Data for Turbulence and Combustion.,”* CISM Inter., vol. 592, T. K. Sengupta and S. Pirozzoli, Eds. Springer International Publishing, 2019.

[3] L. Lestandi, *Numerical Study of Low Rank Approximation Methods for Mechanics Data and its Analysis*, preprint submitted to JCOMP,2020
## Profiling techniques
As CPU time is a pressing issue for many in HPC fields, we provide with this code
simple methods for profiling our methods. It is left to the user discretion to
acquire the listed tools.

3 options are available:
	- _gprof2dot_

> python3 -m profile -o ../profiling/output.pstats analysis/numerical_tests.py && python3 ../../gprof2dot.py -f pstats ../profiling/output.pstats | dot -Tsvg -o ../profiling/call_trace.svg

	- graphviz Nice look, buggy
> pycallgraph graphviz -- ./benchmark_multivariable.py

	- snakeviz Nice looking browser interface.
> python3 -m cProfile -o bench.prof analysis/numerical_tests.py && snakeviz bench.prof
