<h1>Pydecomp Overview</h1>

A new organisation has been applied to the code. Main features are working. Ready for optimization by @Diego !

In this project we try to build an easy to use library for decomposition of
multiparameter Data. It is written in python and aims at being used by
people that do not master the decomposition techniques but want to use it.

Example :
	- industrial partners
	- data compression for labmates

<h1> Profiling techniques </h1>
3 options are available:
	- *gprof2dot*

> python3 -m profile -o ../profiling/output.pstats analysis/numerical_tests.py && python3 ../../gprof2dot.py -f pstats ../profiling/output.pstats | dot -Tsvg -o ../profiling/call_trace.svg

	- graphviz Nice look, buggy
> pycallgraph graphviz -- ./benchmark_multivariable.py

	- snakeviz Nice looking browser interface.
> python3 -m cProfile -o bench.prof analysis/numerical_tests.py && snakeviz bench.prof
