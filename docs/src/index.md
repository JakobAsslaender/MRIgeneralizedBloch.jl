```@meta
CurrentModule = MRIgeneralizedBloch
```

# MRIgeneralizedBloch.jl

Documentation for the [MRIgeneralizedBloch.jl](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl) package, which implements the generalized Bloch model for simulating magnetization transfer (MT), as described in our [paper](https://arxiv.org/pdf/2107.11000.pdf). The key innovation of the model is to generalize the original Bloch model to arbitrary lineshapes, such as the [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404) which has been shown to describe brain white matter well. This enables a more accurate description of the spin dynamics during short RF-pulses compared to previous MT models.

The package allows to simulate the dynamics of an isolated semi-solid spin pool during RF-pulses, as well as the dynamics of a coupled spin system with a free spin pool, for which we use the Bloch model, and a semi-solid pool, which we describe with the generalized Bloch model.

A bare bone demonstration of the interface can found in the *Quick Start Tutorial*, where the Section [Single RF Pulse](@ref) demonstrates the basic and flexible simulation of the spin dynamics during a single RF pulse and Section [Balanced Hybrid-State Free Precession Pulse Sequence](@ref) demonstrates an efficient simulation of a train of RF pulses.

More details on the implementation are provided in the Section *Generalized Bloch Paper*, which reproduces all simulations, data analyses, and figures of the [generalized Bloch paper]((https://arxiv.org/pdf/2107.11000.pdf)):

```@contents
Pages=[
        "build_literate/Greens_functions.md",
        "build_literate/Simulation_ContinuousWave.md",
        "build_literate/Simulation_Pulse.md",
        "build_literate/Analyze_NMR_IR_Data.md",
        "build_literate/Analyze_NMR_PreSat_Data.md",
        "build_literate/Linear_Approximation.md",
]
Depth = 2
```

Section [Non-Linear Least Square Fitting](@ref) demonstrates a simple method for parameter estimation at the example of a *Balanced Hybrid-State Free Precession Pulse Sequence*; and Section [Optimal Control](@ref) outlines the interface for optimizing RF pulse trains for parameter estimation. More details about these topics can be found in the paper [Rapid quantitative magnetization transfer imaging: utilizing the hybrid state and the generalized Bloch model](http://TODO.org).

The documentation of all exported functions can be found in the [API](@ref) Section.