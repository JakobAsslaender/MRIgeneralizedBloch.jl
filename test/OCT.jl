using MRIgeneralizedBloch
using Test
using LinearAlgebra
using StaticArrays
using FiniteDifferences
R2slT = precompute_R2sl()

function calc_CRB(ω1,TRF,w,grad_moment)
    s = calculatesignal_linearapprox(ω1.*TRF, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_list, grad_moment)
    s = reshape(s, size(s,1)*size(s,2), size(s,3))
    return real(w * diag(inv(s' * s)))
end

##
Npulse = 100
α = [π; abs.(π/2 * sin.(π * ((0:Npulse-2) / (Npulse-2)) ))]
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
TRF[1] = 500e-6
ω1 = α ./ TRF
TR = 3.5e-3

grad_moment = [:spoiler_dual; fill(:balanced,length(ω1)-1)]

B1 = 1
ω0 = 0
m0s = 0.25
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rex = 20

grad_list = (grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())
w = transpose([1/m0s;1/R1f;1/R2f;0;0;0;0;0;0].^2)

## ########################################################################
# Test dCRBdm
###########################################################################
# m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_list)
# CRB_fd, d_fd = dCRBdm_fd(m,w)

# m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_list, output=:realmagnetization)
# CRB, d = MRIgeneralizedBloch.dCRBdm(m, w)

# _dCRBdm    = [d(t,r,g)[i]    for t=1:size(m,1), r=1:size(m,2), g=1:size(m,3), i ∈ 1:11]
# _dCRBdm_fd = [d_fd(t,r,g)[i] for t=1:size(m,1), r=1:size(m,2), g=1:size(m,3), i ∈ 1:11]
# # _dCRBdm_fd[isnan.(_dCRBdm_fd)]
# @test _dCRBdm ≈ _dCRBdm_fd rtol = 1e-2


## ########################################################################
# Test OCT gradients: single loop
###########################################################################
# CRB_gradient_OCT: analytical

(F0, grad_ω1, grad_TRF) = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, w; grad_moment)

f_ω1 = _ω1  -> calc_CRB(_ω1, TRF,w,grad_moment)
f_TRF =_TRF -> calc_CRB( ω1,_TRF,w,grad_moment)

_grad_ω1_fd  = grad(central_fdm(5,1; factor=1e6), f_ω1, ω1)[1] # Finite Difference gradient: ω1
@test grad_ω1 ≈ _grad_ω1_fd rtol = 1e-3

_grad_TRF_fd = grad(central_fdm(5,1; factor=1e6, max_range=5e-8), f_TRF, TRF)[1] # Finite Difference gradient: TRF
@test grad_TRF ≈ _grad_TRF_fd rtol = 1e-3


## ########################################################################
# Test OCT gradients: double loop
###########################################################################
α = [α; α]
ω1 = [ω1; ω1]
TRF = [TRF; TRF]
grad_moment = [grad_moment; grad_moment]

##
(F0, grad_ω1_2, grad_TRF_2) = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, w; grad_moment,nSeq=2)
grad_ω1_2 .*= 4
grad_TRF_2 .*= 4

@test grad_ω1 ≈ grad_ω1_2[1:end÷2]       rtol = 1e-5
@test grad_ω1 ≈ grad_ω1_2[end÷2+1:end]   rtol = 1e-5
@test grad_TRF ≈ grad_TRF_2[1:end÷2]     rtol = 1e-5
@test grad_TRF ≈ grad_TRF_2[end÷2+1:end] rtol = 1e-5