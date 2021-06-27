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

export calculatesignal_gbloch_ide
export LinearApprox_calculate_magnetization
export LinearApprox_calculate_signal
export calculatesignal_graham_ode

export apply_hamiltonian_gbloch!
export apply_hamiltonian_sled!
export hamiltonian_linear

export greens_lorentzian
export greens_gaussian
export greens_superlorentzian
export dG_o_dT2s_x_T2s_lorentzian
export dG_o_dT2s_x_T2s_gaussian
export dG_o_dT2s_x_T2s_superlorentzian
export interpolate_greens_function

export precompute_R2sl
export evaluate_R2sl_vector
export calculatesignal_linearapprox

export grad_m0s
export grad_R1
export grad_R2f
export grad_Rx
export grad_T2s
export grad_ω0
export grad_B1

abstract type grad_param end
struct grad_m0s <: grad_param end
struct grad_R1 <: grad_param end
struct grad_R2f <: grad_param end
struct grad_Rx <: grad_param end
struct grad_T2s <: grad_param end
struct grad_ω0 <: grad_param end
struct grad_B1 <: grad_param end

include("DiffEq_Hamiltonians.jl")
include("Linearized_R2s.jl")
include("MatrixExp_Solvers.jl")
include("DiffEq_Sovlers.jl")
include("Greens_functions.jl")
include("MatrixExp_Hamiltonians.jl")

end