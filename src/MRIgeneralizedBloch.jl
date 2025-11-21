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

export precompute_R2sl
export evaluate_R2sl_vector
export calculatesignal_linearapprox

export fit_gBloch
export qMTmap

export grad_m0s
export grad_R1a
export grad_R1f
export grad_R1s
export grad_R2f
export grad_Rex
export grad_T2s
export grad_ω0
export grad_B1

export grad_m0_M
export grad_m0_NM
export grad_m0_MW
export grad_Rx_M_MW
export grad_Rx_MW_IEW
export grad_Rx_IEW_NM
export grad_R1_M
export grad_R1_NM
export grad_R1_IEW
export grad_R1_MW
export grad_R2_MW
export grad_R2_IEW
export grad_T2_M
export grad_T2_NM

abstract type grad_param end
struct grad_m0s <: grad_param end
struct grad_R1a <: grad_param end
struct grad_R1f <: grad_param end
struct grad_R1s <: grad_param end
struct grad_R2f <: grad_param end
struct grad_Rex <: grad_param end
struct grad_T2s <: grad_param end
struct grad_ω0  <: grad_param end
struct grad_B1  <: grad_param end

struct grad_m0_M <: grad_param end
struct grad_m0_NM <: grad_param end
struct grad_m0_MW <: grad_param end
struct grad_Rx_M_MW <: grad_param end
struct grad_Rx_MW_IEW <: grad_param end
struct grad_Rx_IEW_NM <: grad_param end
struct grad_R1_M <: grad_param end
struct grad_R1_NM <: grad_param end
struct grad_R1_IEW <: grad_param end
struct grad_R1_MW <: grad_param end
struct grad_R2_MW <: grad_param end
struct grad_R2_IEW <: grad_param end
struct grad_T2_M <: grad_param end
struct grad_T2_NM <: grad_param end

include("DiffEq_Hamiltonians.jl")
include("Linearized_R2s.jl")
include("MatrixExp_Solvers.jl")
include("DiffEq_Sovlers.jl")
include("Greens_functions.jl")
include("MatrixExp_Hamiltonians.jl")
include("MatrixExp_Hamiltonian_Gradients.jl")
include("Gradient_Hamiltonians.jl")
include("NLLSFit.jl")
include("OptimalControl.jl")
include("OptimalControlHelpers.jl")

include("4Comp_MatrixExp_Hamiltonians.jl")
include("4Comp_MatrixExp_Solvers.jl")
include("4Comp_MatrixExp_Hamiltonian_Gradients.jl")

end