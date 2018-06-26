<h1> Short reminder on profiling </h1>

Three ways avaible to do so:
  - gprof2dot
  > python3 -m profile -o profiling/output.pstats
  > python3 ../gprof2dot.py -f pstats profiling/output.pstats | dot -Tsvg -o profiling/call_trace.svg

  efficient tool, generates a tree view of calls
  - snakeviz
  >python3 -m cProfile -o bench.prof benchmark_multivariable.py && snakeviz bench.prof

  Very easy to use, opens a browser window with two representations

  - graphviz
  >pycallgraph graphviz -- ./mypythonscript.py

  Or, you can profile particular parts of your code:
  ```python
  from pycallgraph import PyCallGraph
  from pycallgraph.output import GraphvizOutput

  with PyCallGraph(output=GraphvizOutput()):
      code_to_profile()
  ```
