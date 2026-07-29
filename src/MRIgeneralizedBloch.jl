module MRIgeneralizedBloch

using QuadGK
using DelayDiffEq
using DifferentialEquations
using OrdinaryDiffEqHighOrderRK
using Interpolations
using ApproxFun
import Cubature
using SpecialFunctions
using StaticArrays
using LinearAlgebra
using NLsolve
using ExponentialUtilities
using LsqFit

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

export simulate_gbloch_ide
export simulate_graham_ode

export precompute_R2sl
export R2slInterpolants
export evaluate_R2sl_vector
export simulate_linearapprox

export fit_gBloch
export qMTmap

# 3 pool
export fit_gBloch_3pool
export qMTparam_3pool
export qMTmap_3pool

export crb_gradient
export bound_omega1_TRF!, get_bounded_omega1_TRF, apply_bounds_to_grad!
export penalty_alpha_curvature!, penalty_RF_power!, penalty_TRF_variation!

export grad_M0
export grad_m0s
export grad_R1a
export grad_R1f
export grad_R1s
export grad_R2f
export grad_Rex
export grad_T2s
export grad_ω0
export grad_B1
# 3-pool gradient parameter types
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

abstract type grad_param end
struct grad_M0  <: grad_param end
struct grad_m0s <: grad_param end
struct grad_R1a <: grad_param end
struct grad_R1f <: grad_param end
struct grad_R1s <: grad_param end
struct grad_R2f <: grad_param end
struct grad_Rex <: grad_param end
struct grad_T2s <: grad_param end
struct grad_ω0  <: grad_param end
struct grad_B1  <: grad_param end
# 3-pool gradient parameter types
struct grad_m0_mm  <: grad_param end
struct grad_m0_rw  <: grad_param end
struct grad_R1_fw  <: grad_param end
struct grad_R1_rw  <: grad_param end
struct grad_R1_mm  <: grad_param end
struct grad_R2_fw  <: grad_param end
struct grad_R2_rw  <: grad_param end
struct grad_T2_mm  <: grad_param end
struct grad_Rx_fw_mm <: grad_param end
struct grad_Rx_rw_fw <: grad_param end
struct grad_Rx_mm_rw <: grad_param end

include("DiffEq_Hamiltonians.jl")
include("Linearized_R2s.jl")
include("MatrixExp_Solvers.jl")
include("DiffEq_Solvers.jl")
include("Greens_Functions.jl")
include("MatrixExp_Hamiltonians.jl")
include("MatrixExp_Hamiltonian_Gradients.jl")
include("NLLSFit.jl")
include("OptimalControl.jl")
include("OptimalControlHelpers.jl")
include("Deprecated.jl")

end