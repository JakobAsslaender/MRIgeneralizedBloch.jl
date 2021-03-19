using LinearAlgebra
using BenchmarkTools
using MAT
using Revise
using Plots
plotlyjs(ticks=:native)
theme(:lime)

using MT_generalizedBloch

## set parameters
ω0 = 0.0
B1 = 1.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TR = 3.5e-3

control = matread(expanduser("examples/control_MT_v3p2_TR3p5ms_discretized.mat"))["control"]
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF

## IDE solution (no Gradients)
s = gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], 2)
plot(TR * 1:length(TRF), s[1,:] ./ (1 - m0s), label="xf / m0f", legend=:topleft)
plot!(TR * 1:length(TRF), s[2,:] ./ (1 - m0s), label="yf / m0f")
plot!(TR * 1:length(TRF), s[3,:] ./ (1 - m0s), label="zf / m0f")
plot!(TR * 1:length(TRF), s[4,:] ./ m0s, label="zs / m0s")

## Graham's solution
s = Graham_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], 2)
plot!(TR * 1:length(TRF), s[1,:] ./ (1 - m0s), label="G: xf / m0f")
plot!(TR * 1:length(TRF), s[2,:] ./ (1 - m0s), label="G: yf / m0f")
plot!(TR * 1:length(TRF), s[3,:] ./ (1 - m0s), label="G: zf / m0f")
plot!(TR * 1:length(TRF), s[4,:] ./ m0s, label="G: zs / m0s")

##
s = Graham_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, 2)
plot(TR * 1:length(TRF), real(s), label="real(s)", legend=:topleft)
plot!(TR * 1:length(TRF), imag(s), label="imag(s)", legend=:topleft)


## Matrix Exp solvers
TRF = [100e-6; control[1:end - 1,2]]
TRF[733] /= 10
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF

B1 = 1.2
T2s = 10e-6
Rrf_T = PreCompute_Saturation_gBloch(minimum(TRF), maximum(TRF), T2s, T2s, minimum(ω1), maximum(ω1), B1, B1)
# Rrf_T = PreCompute_Saturation_Graham(minimum(TRF), maximum(TRF), T2s, T2s)

s = MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Rrf_T)

plot(s[1,:] ./ (1 - m0s), label="xf / m0f", legend=:topleft)
plot!(s[2,:] ./ (1 - m0s), label="yf / m0f")
plot!(s[3,:] ./ (1 - m0s), label="zf / m0f")
plot!(s[4,:] ./ m0s, label="zs / m0s")

##
TRF = [100e-6; control[1:end - 1,2]]
TRF .= 1e-3
TRF[100:5:200] .= 50e-6
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
α .= 0.8π
ω1 = α ./ TRF

B1 = 1.0
T2s = 10e-6
Rrf_gB = PreCompute_Saturation_gBloch(minimum(TRF), maximum(TRF), T2s, T2s, minimum(ω1), maximum(ω1), B1, B1)
Rrf_Gr = PreCompute_Saturation_Graham(minimum(TRF), maximum(TRF), T2s, T2s)

sgB = MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Rrf_gB)
sGr = MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Rrf_Gr)

plot(sgB[1,:] ./ (1 - m0s), label="gB xf", legend=:bottomright)
plot!(sgB[3,:] ./ (1 - m0s), label="gB zf")
plot!(sgB[4,:] ./ m0s, label="gB zs")

plot!(sGr[1,:] ./ (1 - m0s), label="Gr xf")
plot!(sGr[3,:] ./ (1 - m0s), label="Gr zf")
plot!(sGr[4,:] ./ m0s, label="Gr zs")

plot(sgB[1,:] ./ sGr[1,:])
