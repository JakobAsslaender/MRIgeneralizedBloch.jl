using MRIgeneralizedBloch
using BenchmarkTools
using MAT
using Test

## set parameters
ω0 = 0.0
B1 = 1.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TR = 3.5e-3

control = matread(expanduser("../examples/control_MT_v3p2_TR3p5ms_discretized.mat"))["control"]
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF

print("Time for the R2sl pre-computation: ")
R2s_T = @time PreCompute_Saturation_gBloch(minimum(TRF), maximum(TRF), T2s, T2s, minimum(α), maximum(α), B1, B1)

## ##################################################################
# magnetization functions
#####################################################################
m_gBloch = gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], 2)

## test linear approximation 
m_linapp = MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2s_T)

@test m_gBloch[1,:] ≈ m_linapp[1,:] rtol = 1e-3
@test m_gBloch[2,:] ≈ m_linapp[2,:] atol = 1e-9
@test m_gBloch[3,:] ≈ m_linapp[3,:] rtol = 1e-3
@test m_gBloch[4,:] ≈ m_linapp[5,:] rtol = 1e-3

## test Graham's solution
m_Graham = Graham_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], 2)

@test m_gBloch[1,:] ≈ m_Graham[1,:] rtol = 1e-2
@test m_gBloch[2,:] ≈ m_Graham[2,:] rtol = 1e-2
@test m_gBloch[3,:] ≈ m_Graham[3,:] rtol = 1e-2
@test m_gBloch[4,:] ≈ m_Graham[4,:] rtol = 1e-2

## ##################################################################
# signal functions
#####################################################################
print("Time to solve the full IDE:            ")
s_gBloch = @btime gBloch_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, 2)

print("Time to solve Graham's approximation:  ")
s_Graham = @btime Graham_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, 2)

print("Time to solve the linear approximation:")
s_linapp = @btime MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2s_T)

@test real(s_gBloch) ≈ real(s_linapp) rtol = 1e-3
@test imag(s_gBloch) ≈ imag(s_linapp) atol = 1e9

@test real(s_gBloch) ≈ real(s_Graham) rtol = 1e-2
@test imag(s_gBloch) ≈ imag(s_Graham) atol = 1e-9
