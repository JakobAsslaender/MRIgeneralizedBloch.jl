module MRIgeneralizedBloch

using QuadGK
using DifferentialEquations
using ApproxFun
import Cubature
using SpecialFunctions
using StaticArrays
using LinearAlgebra
using NLsolve

export gBloch_calculate_magnetization
export gBloch_calculate_signal
export LinearApprox_calculate_magnetization
export LinearApprox_calculate_signal
export Graham_calculate_magnetization
export Graham_calculate_signal

export Greens_Lorentzian
export Greens_Gaussian
export Greens_superLorentzian
export dG_o_dT2s_x_T2s_superLorentzian
export interpolate_Greens_Function

export PreCompute_Saturation_gBloch
export PreCompute_Saturation_Graham

export MatrixApprox_calculate_magnetization
export MatrixApprox_calculate_signal
export Calculate_Saturation_rate

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

include("MT_Hamiltonians.jl")
include("MT_Diff_Equation_Sovlers.jl")
include("MT_Exponential_Solvers.jl")
include("Greens_functions.jl")

end