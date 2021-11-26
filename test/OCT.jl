using MRIgeneralizedBloch
using Test
using LinearAlgebra
using StaticArrays
include("OCT_finite_difference_gradients.jl")

##
Npulse = 100
α = abs.(π/2 * sin.(π/2 * (1:Npulse) / Npulse))
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
α[1] = π
TRF[1] = 500e-6
ω1 = α ./ TRF

R2slT = precompute_R2sl(ω1_max = 1.1 * maximum(ω1))
TR = 3.5e-3

B1 = 1
ω0 = 0
m0s = 0.25
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 20

grad_list = [grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]
w = transpose([1/m0s;1/R1f;1/R2f;0;0;0;0;0;0].^2)

## ########################################################################
# Test dCRBdm
###########################################################################
# m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT; grad_list=grad_list)
# CRB_fd, d_fd = dCRBdm_fd(m,w)

# m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT; grad_list=grad_list, output=:realmagnetization)
# CRB, d = MRIgeneralizedBloch.dCRBdm(m, w)

# _dCRBdm    = [d(t,r,g)[i]    for t=1:size(m,1), r=1:size(m,2), g=1:size(m,3), i ∈ 1:11]
# _dCRBdm_fd = [d_fd(t,r,g)[i] for t=1:size(m,1), r=1:size(m,2), g=1:size(m,3), i ∈ 1:11]
# # _dCRBdm_fd[isnan.(_dCRBdm_fd)]
# @test _dCRBdm ≈ _dCRBdm_fd rtol = 1e-2


## ########################################################################
# Test OCT gradients
###########################################################################
# OCT_gradient: analytical
(F0, grad_ω1, grad_TRF) = MRIgeneralizedBloch.OCT_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, w)

## Finite Difference gradient: ω1
_grad_ω1_fd = grad_ω1_fd(w, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list)
@test grad_ω1 ≈ _grad_ω1_fd rtol = 1e-1

## Finite Difference gradient: TRF
_grad_TRF_fd = grad_TRF_fd(w, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list)
@test grad_TRF ≈ _grad_TRF_fd rtol = 1e-1


## OCT_TV_gradient: analytical
λ = 1
(F0, grad_ω1, grad_TRF) = MRIgeneralizedBloch.OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, w, λ)

## FD gradient: ω1
_,_grad_ω1_fd = grad_TV_ω1_fd(w, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, λ)
@test grad_ω1 ≈ _grad_ω1_fd rtol = 1e-1