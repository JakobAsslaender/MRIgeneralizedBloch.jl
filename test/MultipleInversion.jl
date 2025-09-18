using MRIgeneralizedBloch
using Test
using LinearAlgebra
using StaticArrays
R2slT = precompute_R2sl()

##
Npulse = 100
α = abs.(π/2 * sin.(π/2 * ((0:Npulse-1) / Npulse .+ 0.5)))
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
α[1] = π
TRF[1] = 500e-6
grad_moment = [:crusher; fill(:balanced, length(α)-1)]
ω1 = α ./ TRF
TR = 3.5e-3

B1 = 1
ω0 = 0
m0s = 0.25
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
K = 20
nTR = 0.07

grad_list = (grad_m0s(), grad_R1f(), grad_R2f(), grad_K(), grad_nTR(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())
w = transpose([1/m0s;1/R1f;1/R2f;0;0;0;0;0;0].^2)


s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT; grad_list, grad_moment)

## ########################################################################
# simulate two loops
###########################################################################
s2 = calculatesignal_linearapprox([α; α], [TRF; TRF], TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT; grad_list, grad_moment=[grad_moment; grad_moment])

@test s1 ≈ s2[1:end÷2,:,:]
@test s1 ≈ s2[end÷2+1:end,:,:]

## ########################################################################
# simulate tree loops
###########################################################################
s3 = calculatesignal_linearapprox([α; α; α], [TRF; TRF; TRF], TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT; grad_list, grad_moment=[grad_moment; grad_moment; grad_moment])

@test s1 ≈ s3[1:end÷3,:,:]
@test s1 ≈ s3[end÷3+1:2*end÷3,:,:]
@test s1 ≈ s3[2*end÷3+1:end,:,:]