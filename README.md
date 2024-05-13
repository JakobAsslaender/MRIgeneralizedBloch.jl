# MRIgeneralizedBloch.jl


| **Documentation**         | **Generalized Bloch Paper**   | **In Vivo qMT Paper**         | **unconstrained T<sub>1</sub> Paper** | **Build Status**                      |
|:------------------------- |:------------------------------|:----------------------------- |:------------------------------------- |:------------------------------------- |
| [![][docs-img]][docs-url] | [![][paper-img1]][paper-url1] | [![][paper-img2]][paper-url2] | [![][paper-img3]][paper-url3]         | [![][gh-actions-img]][gh-actions-url] |
|                           | [![][arXiv-img1]][arXiv-url1] | [![][arXiv-img2]][arXiv-url2] | [![][arXiv-img3]][arXiv-url3]         | [![][codecov-img]][codecov-url]       |


MRIgeneralizedBloch.jl is a Julia package that implements the [generalized Bloch equations](https://doi.org/10.1002/mrm.29071) for modeling the dynamics of the semi-solid spin pool in magnetic resonance imaging (MRI), and its exchange with the free spin pool. It utilizes the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package to solve its integro-differential equation. It also implements a linear approximation of the generalized Bloch equations that assumes rectangular radio frequency pulses and uses matrix exponentiation of [static arrays](https://github.com/JuliaArrays/StaticArrays.jl), which results in virtually non-allocating and extremely fast code. 

For more details, please refer to the [paper](https://doi.org/10.1002/mrm.29071) and the above linked documentation, which also contains scripts that reproduce all figures in the paper.


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl/stable

[gh-actions-img]: https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/actions

[codecov-img]: https://codecov.io/gh/JakobAsslaender/MRIgeneralizedBloch.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JakobAsslaender/MRIgeneralizedBloch.jl

[arXiv-img1]: https://img.shields.io/badge/arXiv-2107.11000-blue.svg
[arXiv-url1]: https://arxiv.org/pdf/2107.11000.pdf

[arXiv-img2]: https://img.shields.io/badge/arXiv-2207.08259-blue.svg
[arXiv-url2]: https://arxiv.org/pdf/2207.08259.pdf

[arXiv-img3]: https://img.shields.io/badge/arXiv-2301.08394-blue.svg
[arXiv-url3]: https://arxiv.org/pdf/2301.08394.pdf

[paper-img1]: https://img.shields.io/badge/doi-10.1002/mrm.29071-blue.svg
[paper-url1]: https://doi.org/10.1002/mrm.29071

[paper-img2]: https://img.shields.io/badge/doi-10.1002/mrm.29951-blue.svg
[paper-url2]: https://doi.org/10.1002/mrm.29951

[paper-img3]: https://img.shields.io/badge/doi-10.1162/imag_a_00177-blue.svg
[paper-url3]: https://doi.org/10.1162/imag_a_00177
