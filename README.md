# Net-Trees

The input to this problem is a set of points in a general metric space.
We organize the points into a data structure called *net-tree* so that it allows efficient proximity queries for metrics with bounded dimension (doubling metrics).
This repository contains a Python implementation of semi-compressed net-trees.

## Performance Notes

Distance computations in `Euclidean`, `Manhattan`, and `LInfinity` metrics now support optional Numba acceleration.

- If `numba` and `numpy` are installed, distance kernels are JIT-compiled automatically.
- If either dependency is missing, the code falls back to the original pure-Python implementations.
- Set environment variable `NETTREES_DISABLE_NUMBA=1` to force pure-Python distance code.

Here is the link to our paper:
<https://arxiv.org/abs/1809.01308>

[![DOI](https://zenodo.org/badge/147385418.svg)](https://zenodo.org/badge/latestdoi/147385418)
