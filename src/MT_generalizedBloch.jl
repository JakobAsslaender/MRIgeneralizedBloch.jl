module MT_generalizedBloch

using QuadGK
using DifferentialEquations
using ApproxFun
import Cubature
using SpecialFunctions
using StaticArrays
using LinearAlgebra

export gBloch_calculate_magnetization
export gBloch_calculate_signal
export LinearApprox_calculate_magnetization
export LinearApprox_calculate_signal
export Graham_calculate_magnetization
export Graham_calculate_signal

export PreCompute_Saturation_gBloch
export PreCompute_Saturation_Graham

export MatrixApprox_calculate_magnetization
export MatrixApprox_calculate_signal
export Calculate_Saturation_rate

export gBloch_Hamiltonian!
export gBloch_Hamiltonian_Gradient!
export gBloch_Hamiltonian_ApproxFun!
export gBloch_Hamiltonian_Gradient_ApproxFun!
export FreePrecession_Hamiltonian!
export Linear_Hamiltonian!
export Linear_Hamiltonian_InversionPulse!
export Graham_Hamiltonian!
export Sled_Hamiltonian!

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

end