using MRIgeneralizedBloch
using Test
using LinearAlgebra
using StaticArrays
using FiniteDifferences
R2slT = precompute_R2sl()

function calc_CRB(ω1, TRF, w, grad_moment)
    s = calculatesignal_linearapprox(ω1 .* TRF, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_list, grad_moment)
    s = reshape(s, size(s, 1) * size(s, 2), size(s, 3))
    return real(w * diag(inv(s' * s)))
end

##
Npulse = 100
α = [π; abs.(π / 2 * sin.(π * ((0:Npulse-2) / (Npulse - 2))))]
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
TRF[1] = 500e-6
ω1 = α ./ TRF
TR = 3.5e-3

grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(ω1)]

B1 = 1
ω0 = 0
m0s = 0.25
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rex = 20

grad_list = (grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())
w = transpose([1 / m0s, 1 / R1f, 1 / R2f, 0, 0, 0, 0, 0, 0] .^ 2)

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

_, _, _ = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, w) # test default grad_moment
F0, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, w; grad_moment)

f_ω1 = _ω1 -> calc_CRB(_ω1, TRF, w, grad_moment)
f_TRF = _TRF -> calc_CRB(ω1, _TRF, w, grad_moment)

_grad_ω1_fd = grad(central_fdm(5, 1; factor=1e6), f_ω1, ω1)[1] # Finite Difference gradient: ω1
@test grad_ω1 ≈ _grad_ω1_fd rtol = 1e-3

_grad_TRF_fd = grad(central_fdm(5, 1; factor=1e6, max_range=5e-8), f_TRF, TRF)[1] # Finite Difference gradient: TRF
@test grad_TRF ≈ _grad_TRF_fd rtol = 1e-3


## ########################################################################
# Test OCT gradients: double loop
###########################################################################
α = hcat(α, α)
ω1 = hcat(ω1, ω1)
TRF = hcat(TRF, TRF)
grad_moment = hcat(grad_moment, grad_moment)

##
F0, grad_ω1_2, grad_TRF_2 = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, w; grad_moment)
grad_ω1_2 .*= 4
grad_TRF_2 .*= 4

@test grad_ω1 ≈ grad_ω1_2[1:end÷2] rtol = 1e-5
@test grad_ω1 ≈ grad_ω1_2[end÷2+1:end] rtol = 1e-5
@test grad_TRF ≈ grad_TRF_2[1:end÷2] rtol = 1e-5
@test grad_TRF ≈ grad_TRF_2[end÷2+1:end] rtol = 1e-5


## ########################################################################
# Test helper functions
###########################################################################
ω1_min=fill(π/2e-3, size(ω1))
ω1_max=fill(π/1e-3, size(ω1))
TRF_min=fill(200e-6, size(TRF))
TRF_max=fill(400e-6, size(TRF))

_ω1 = copy(ω1)
_TRF = copy(TRF)

x = MRIgeneralizedBloch.bound_ω1_TRF!(_ω1, _TRF; ω1_min, ω1_max, TRF_min, TRF_max)
ω1_b, TRF_b = MRIgeneralizedBloch.get_bounded_ω1_TRF(x; NSeq=2, ω1_min, ω1_max, TRF_min, TRF_max)

@test ω1_b ≈ _ω1
@test TRF_b ≈ _TRF
@test all(ω1_b .<= ω1_max)
@test all(ω1_b .>= ω1_min)
@test all(TRF_b .<= TRF_max)
@test all(TRF_b .>= TRF_min)

##
ω1_min=zeros(size(ω1))
ω1_max=fill(1e6, size(ω1))
TRF_min=fill(0, size(TRF))
TRF_max=fill(1, size(TRF))

_ω1 = copy(ω1)
_TRF = copy(TRF)

x = MRIgeneralizedBloch.bound_ω1_TRF!(_ω1, _TRF; ω1_min, ω1_max, TRF_min, TRF_max)
ω1_b, TRF_b = MRIgeneralizedBloch.get_bounded_ω1_TRF(x; NSeq=2, ω1_min, ω1_max, TRF_min, TRF_max)

@test ω1_b ≈ ω1
@test TRF_b ≈ TRF
@test ω1_b ≈ _ω1
@test TRF_b ≈ _TRF
@test all(ω1_b .<= ω1_max)
@test all(ω1_b .>= ω1_min)
@test all(TRF_b .<= TRF_max)
@test all(TRF_b .>= TRF_min)

##
function fg!(F, G, x)
    ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x; NSeq=2, ω1_min, ω1_max, TRF_min, TRF_max)

    F, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, w; grad_moment)
    F = abs(F)

    F += MRIgeneralizedBloch.second_order_α!(grad_ω1, grad_TRF, ω1, TRF; grad_moment, λ=1e4)
    F += MRIgeneralizedBloch.RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1e-3, Pmax=3e6, TR)
    F += MRIgeneralizedBloch.TRF_TV!(grad_TRF, TRF; grad_moment, λ=1e3)

    MRIgeneralizedBloch.apply_bounds_to_grad!(G, x, grad_ω1, grad_TRF; ω1_min, ω1_max, TRF_min, TRF_max)
    return F
end

G = similar(x)
G_fd = grad(central_fdm(5, 1; factor=1e6), x -> fg!(nothing, G, x), x)[1] # Finite Difference gradient: TRF

F = fg!(nothing, G, x)

@test G ≈ G_fd rtol = 1e-2