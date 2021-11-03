## TODO: Write and include tests
using Test

##
grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]
m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT; grad_list=grad_list)
m = reshape(m, (size(m,1)*size(m,2), size(m,3)))

function dLdy_fd(m, w)
    Δm = 1e-9
    CRB1 = w * diag(inv(m' * m))
    _dLdy = similar(m)
    for i in eachindex(m)
        dm = copy(m)
        dm[i] = m[i] + Δm 
        _dLdy[i] = real(CRB1 - w * diag(inv(dm' * dm))) / Δm

        dm = copy(m)
        dm[i] = m[i] + 1im * Δm 
        _dLdy[i] += 1im * real(CRB1 - w * diag(inv(dm' * dm))) / Δm
    end
    return _dLdy    
end


w = transpose([1/m0s;1/R1;1/R2f;0;0;0;0;0].^2)
_dLdy_fd = dLdy_fd(m,w)
_dLdy = MRIgeneralizedBloch.dLdy(m, w)

@test _dLdy ≈ _dLdy_fd rtol = 1e-3

@test _dLdy[:,1] ≈ _dLdy_fd[:,1] rtol = 1e-3
@test _dLdy[:,2] ≈ _dLdy_fd[:,2] rtol = 1e-2
@test _dLdy[:,3] ≈ _dLdy_fd[:,3] rtol = 1e-3
@test _dLdy[:,4] ≈ _dLdy_fd[:,4] rtol = 1e-3
@test _dLdy[:,5] ≈ _dLdy_fd[:,5] rtol = 1e-3
@test _dLdy[:,6] ≈ _dLdy_fd[:,6] rtol = 1e-2
@test _dLdy[:,7] ≈ _dLdy_fd[:,7] rtol = 1e-2
@test _dLdy[:,8] ≈ _dLdy_fd[:,8] rtol = 1e-2


##
plot(real.(_dLdy))
plot!(real.(_dLdy_fd))

##
plot(imag.(_dLdy))
plot!(imag.(_dLdy_fd))


#####################################################################################################\
## OCT gradients
using MRIgeneralizedBloch
using Plots

##
TR = 3.5e-3
R2slT = precompute_R2sl(50e-6, 750e-6, 8e-6, 12e-6, 0, π, 0.6, 1.2)

## 
m0s = 0.1
R1 = 1
R2f = 1 / 0.065
T2s = 10e-6
Rx = 40
ω0 = 0
B1 = 1
# weights = transpose([0, 1 / m0s, 1 / R1, 1 / R2f, 0, 0, 0, 0].^2)
# grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]
weights = transpose([0, 1 / R1, 1 / R2f].^2)
grad_list = [grad_R1(), grad_R2f()]
  
## initialize
ω1  = 1e3 * ones(100)
TRF = 100e-6 .+ 700e-6 * ones(100)
ω1[1]*TRF[1]/pi*180

## OCT_gradient: analytical
(F0, grad_ω1, grad_TRF) = MRIgeneralizedBloch.OCT_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights)

## OCT_gradient: ω1
Δω1 = 1e-2
grad_ω1_fd = similar(grad_ω1)
for i ∈ eachindex(ω1)
    ω1[i] += Δω1
    (F1, _, _) = MRIgeneralizedBloch.OCT_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights)
    ω1[i] -= Δω1

    grad_ω1_fd[i] = (F1 - F0) / Δω1
end
grad_ω1_fd[1] = 0

plot(grad_ω1)
plot!(grad_ω1_fd)

##
plot(grad_ω1 .- grad_ω1_fd)

## OCT_gradient: TRF
ΔTRF = 1e-8
grad_TRF_fd = similar(grad_TRF)
for i ∈ eachindex(ω1)
    TRF[i] += ΔTRF
    (F1, _, _) = MRIgeneralizedBloch.OCT_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights)
    TRF[i] -= ΔTRF

    grad_TRF_fd[i] = (F1 - F0) / ΔTRF
end
grad_TRF_fd[1] = 0

plot(grad_TRF)
plot!(grad_TRF_fd)

##
plot(grad_TRF ./ grad_TRF_fd)

##########################################################
## OCT_TV_gradient: analytical
λ = 1
weights .= 0
(F0, grad_ω1, grad_TRF) = MRIgeneralizedBloch.OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights, λ)

## OCT_TV_gradient: ω1
Δω1 = 1e-2
grad_ω1_fd = similar(grad_ω1)
for i ∈ eachindex(ω1)
    ω1[i] += Δω1
    (F1, _, _) = MRIgeneralizedBloch.OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights, λ)
    ω1[i] -= Δω1

    grad_ω1_fd[i] = (F1 - F0) / Δω1
end
grad_ω1_fd[1] = 0

plot(grad_ω1)
plot!(grad_ω1_fd)

##
plot(grad_ω1 ./ grad_ω1_fd)

## OCT_TV_gradient: TRF
ΔTRF = 1e-8
grad_TRF_fd = similar(grad_TRF)
for i ∈ eachindex(ω1)
    TRF[i] += ΔTRF
    (F1, _, _) = MRIgeneralizedBloch.OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights, λ)
    TRF[i] -= ΔTRF

    grad_TRF_fd[i] = (F1 - F0) / ΔTRF
end
grad_TRF_fd[1] = 0

plot(grad_TRF)
plot!(grad_TRF_fd)

##
plot(grad_TRF ./ grad_TRF_fd)


#####################################################################################################
using MRIgeneralizedBloch
using ExponentialUtilities
using LinearAlgebra
using StaticArrays
using MAT
using BenchmarkTools
using Revise
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); # hide

## set parameters
ω0 = 300
B1 = 0.7
m0s = 0.15
R1 = 1
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30
TR = 3.5e-3

control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../HSFPstuff/control_MT_v3p2_TR3p5ms_discretized.mat")))["control"]
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
rfphase_increment = [π]
R2slT = precompute_R2sl(minimum(TRF) - 1e-6, maximum(TRF) + 1e-6, 7e-6, 13e-6, minimum(α), maximum(α), 0.6, 1.3)
(R2sl, dR2sldT2s, dR2sldB1, dR2sldω1, dR2sldTRF, dR2sldT2sdω1, dR2sldB1dω1) = R2slT


##
TRF = TRF[1:100]
α = α[1:100]
ω1 = α ./ TRF

##
grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]
weights = transpose([0; 1 / m0s;1 / R1;1 / R2f;0; 0; 0; 0].^2)
# grad_list = [grad_m0s(), grad_R1(), grad_ω0()]
# weights = transpose([1; 1; 1; 1].^2)

## #################################################################
include("OCT_FD_tmp.jl")
_grad_ω1_fd  = grad_ω1_fd(weights, ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list)
_grad_TRF_fd = grad_TRF_fd(weights, ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list)

## #################################################################
(CRB, grad_ω1, grad_TRF) = MRIgeneralizedBloch.OCT_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights)
# @benchmark MRIgeneralizedBloch.OCT_gradient($ω1, $TRF, $TR, $ω0, $B1, $m0s, $R1, $R2f, $Rx, $T2s, $R2slT, $grad_list, $weights)


##
plot(_grad_ω1_fd, label="fd")
plot!(grad_ω1, label="OCT")

##
plot(_grad_ω1_fd .- grad_ω1)

##
plot(_grad_TRF_fd, label="fd")
plot!(grad_TRF, label="OCT")

##
plot(_grad_TRF_fd ./ grad_TRF)