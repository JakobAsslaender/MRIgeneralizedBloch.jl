module MRIgeneralizedBloch

using QuadGK
using DifferentialEquations
using Interpolations
using ApproxFun
import Cubature
using SpecialFunctions
using StaticArrays
using LinearAlgebra
using NLsolve
using ExponentialUtilities
using LsqFit
using LinearAlgebra

export apply_hamiltonian_gbloch!
export apply_hamiltonian_linear!
export apply_hamiltonian_graham_superlorentzian!
export graham_saturation_rate_spectral
export graham_saturation_rate_single_frequency
export apply_hamiltonian_sled!
export hamiltonian_linear
export d_hamiltonian_linear_dω1

export greens_lorentzian
export greens_gaussian
export greens_superlorentzian
export lineshape_lorentzian
export lineshape_gaussian
export lineshape_superlorentzian
export dG_o_dT2s_x_T2s_lorentzian
export dG_o_dT2s_x_T2s_gaussian
export dG_o_dT2s_x_T2s_superlorentzian
export interpolate_greens_function

export calculatesignal_gbloch_ide
export calculatesignal_graham_ode

export precompute_R2_mm_l
export evaluate_R2_mm_l_vector
export calculatesignal_linearapprox

export fit_gBloch
export qMTmap

export grad_m0_mm
export grad_m0_rw
export grad_R1_fw
export grad_R1_rw
export grad_R1_mm
export grad_R2_fw
export grad_R2_rw
export grad_T2_mm
export grad_Rx_fw_mm
export grad_Rx_rw_fw
export grad_Rx_mm_rw
export grad_ω0
export grad_B1

# include("DiffEq_Hamiltonians.jl") # ToDo
include("Linearized_R2_mm.jl")
include("MatrixExp_Solvers.jl")
include("DiffEq_Sovlers.jl")
include("Greens_functions.jl")
include("MatrixExp_Hamiltonians.jl")
include("MatrixExp_Hamiltonian_Gradients.jl")
include("NLLSFit.jl")
include("OptimalControl.jl")
include("OptimalControlHelpers.jl")
include("grad_param.jl")

end