# MRIgeneralizedBloch.jl


| **Documentation**         | **Build Status**                                                      |
|:------------------------- |:--------------------------------------------------------------------- |
| [![][docs-img]][docs-url] | [![][gh-actions-img]][gh-actions-url] [![][codecov-img]][codecov-url] |

MRIgeneralizedBloch.jl is a Julia package that implements the generalized Bloch equations for modeling the dynamics of the semi-solid spin pool in magnetic resonance imaging (MRI), and its exchange with the free spin pool. It utilizes the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package to solve integro-differential equation. It also implements a linear approximation of the generalized Bloch equations that assumes rectangular radio frequency pulses and uses matrix exponentiation of [static arrays](https://github.com/JuliaArrays/StaticArrays.jl), which results in almost non-allocating and extremely fast code. 

For more details and scripts that reproduce all figures in the paper, please refer to above linked documentation. 


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl/stable)
 [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl/dev)
 
[![Build Status](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/workflows/CI/badge.svg)](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/actions)
[![Coverage](https://codecov.io/gh/JakobAsslaender/MRIgeneralizedBloch.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JakobAsslaender/MRIgeneralizedBloch.jl)


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl/dev

[gh-actions-img]: https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/actions

[codecov-img]: https://codecov.io/gh/JakobAsslaender/MRIgeneralizedBloch.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JakobAsslaender/MRIgeneralizedBloch.jl