using MRIgeneralizedBloch
using Test
using LinearAlgebra
using StaticArrays

##
Npulse = 100
α = abs.(π/2 * sin.(π/2 * ((0:Npulse-1) / Npulse .+ 0.5)))
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
α[1] = π
TRF[1] = 500e-6
isInversionPulse = [true; falses(length(α)-1)]
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
# simulate one loop
###########################################################################
s1d = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list)

s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list, isInversionPulse = isInversionPulse)

@test s1 ≈ s1d

## ########################################################################
# simulate two loops
###########################################################################
s2 = calculatesignal_linearapprox([α; α], [TRF; TRF], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list, isInversionPulse = [isInversionPulse; isInversionPulse])

@test s1 ≈ s2[1:end÷2,:,:]
@test s1 ≈ s2[end÷2+1:end,:,:]

## ########################################################################
# simulate tree loops
###########################################################################
s3 = calculatesignal_linearapprox([α; α; α], [TRF; TRF; TRF], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list, isInversionPulse = [isInversionPulse; isInversionPulse; isInversionPulse])

@test s1 ≈ s3[1:end÷3,:,:]
@test s1 ≈ s3[end÷3+1:2*end÷3,:,:]
@test s1 ≈ s3[2*end÷3+1:end,:,:]