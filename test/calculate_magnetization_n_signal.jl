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

grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]

control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../examples/control_MT_v3p2_TR3p5ms_discretized.mat")))["control"]

TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF

## Pre-compute and evalute R2sf
print("Time for the R2sl pre-computation: ")
R2slT = @time precompute_R2sl(minimum(TRF), maximum(TRF), T2s, T2s, minimum(α), maximum(α), B1, B1)

print("Time to evaluate the interpolated R2sl w/o gradients: ")
R2s_vec = @btime evaluate_R2sl_vector($α, $TRF, $B1, $T2s, $R2slT, $[])

print("Time to evaluate the interpolated R2sl with gradients: ")
R2s_vec = @btime evaluate_R2sl_vector($α, $TRF, $B1, $T2s, $R2slT, $grad_list)

## ##################################################################
# magnetization functions
#####################################################################
m_gBloch = calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s; output=:realmagnetization)

# test Graham's solution
m_Graham = calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s; output=:realmagnetization)

@test m_gBloch[:,1] ≈ m_Graham[:,1] rtol = 1e-2
@test m_gBloch[:,2] ≈ m_Graham[:,2] rtol = 1e-2
@test m_gBloch[:,3] ≈ m_Graham[:,3] rtol = 1e-2
@test m_gBloch[:,4] ≈ m_Graham[:,4] rtol = 1e-2

# test linear approximation 
m_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT; output=:realmagnetization)
m_linapp = [m_linapp[i][j] for i=1:size(m_linapp,1), j=1:5]

@test m_gBloch[:,1] ≈ m_linapp[:,1] rtol = 1e-3
@test m_gBloch[:,2] ≈ m_linapp[:,2] atol = 1e-9
@test m_gBloch[:,3] ≈ m_linapp[:,3] rtol = 1e-3
@test m_gBloch[:,4] ≈ m_linapp[:,5] rtol = 1e-3


## ##################################################################
# signal functions w/o gradients
#####################################################################
println("w/o gradients:")
print("Time to solve the full IDE:            ")
s_gBloch = @btime calculatesignal_gbloch_ide($α, $TRF, $TR, $ω0, $B1, $m0s, $R1, $R2f, $Rx, $T2s)

# test Graham's solution
print("Time to solve Graham's approximation:  ")
s_Graham = @btime calculatesignal_graham_ode($α, $TRF, $TR, $ω0, $B1, $m0s, $R1, $R2f, $Rx, $T2s)

@test real(s_gBloch) ≈ real(s_Graham) rtol = 1e-2
@test imag(s_gBloch) ≈ imag(s_Graham) atol = 1e-9

# test linear approximation
print("Time to solve the linear approximation:")
s_linapp = @btime vec(calculatesignal_linearapprox($α, $TRF, $TR, $ω0, $B1, $m0s, $R1, $R2f, $Rx, $T2s, $R2slT))

@test real(s_gBloch) ≈ real(s_linapp) rtol = 1e-3
@test imag(s_gBloch) ≈ imag(s_linapp) atol = 1e-9


## ##################################################################
# signal functions with gradients
#####################################################################
println("with all (8) gradients:")
print("Time to solve the full IDE:            ")
s_gBloch_grad = @time calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list=grad_list)

# test Graham's solution
print("Time to solve Graham's approximation:  ")
s_Graham_grad = @time calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list=grad_list)

@test s_gBloch_grad[:,1] ≈ s_Graham_grad[:,1] rtol = 1e-1
@test s_gBloch_grad[:,2] ≈ s_Graham_grad[:,2] rtol = 1e-1
@test s_gBloch_grad[:,3] ≈ s_Graham_grad[:,3] rtol = 1e-1
@test s_gBloch_grad[:,4] ≈ s_Graham_grad[:,4] rtol = 1e-1
@test s_gBloch_grad[:,5] ≈ s_Graham_grad[:,5] rtol = 1e-1
@test s_gBloch_grad[:,6] ≈ s_Graham_grad[:,6] rtol = 1e-1
@test s_gBloch_grad[:,7] ≈ s_Graham_grad[:,7] rtol = 1e-1
@test s_gBloch_grad[:,8] ≈ s_Graham_grad[:,8] rtol = 1e-1

# test linear approximation
print("Time to solve the linear approximation:")
s_linapp_grad = @time calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT; grad_list=grad_list)
s_linapp_grad = dropdims(s_linapp_grad, dims=2)

@test s_gBloch_grad[:,1] ≈ s_linapp_grad[:,1] rtol = 1e-3
@test s_gBloch_grad[:,2] ≈ s_linapp_grad[:,2] rtol = 1e-3
@test s_gBloch_grad[:,3] ≈ s_linapp_grad[:,3] rtol = 1e-3
@test s_gBloch_grad[:,4] ≈ s_linapp_grad[:,4] rtol = 1e-3
@test s_gBloch_grad[:,5] ≈ s_linapp_grad[:,5] rtol = 1e-3
@test s_gBloch_grad[:,6] ≈ s_linapp_grad[:,6] rtol = 1e-2
@test s_gBloch_grad[:,7] ≈ s_linapp_grad[:,7] rtol = 1e-3
@test s_gBloch_grad[:,8] ≈ s_linapp_grad[:,8] rtol = 1e-3