using DifferentialEquations
using Test
using MRIgeneralizedBloch
using MRIgeneralizedBloch: hamiltonian_linear, propagator_linear_inversion_pulse

## 
max_error = 1e-5

## set parameters
α  = π
ω1 = α / 500e-6
ω0 = 200.0
B1 = 1.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TRF = 500e-6
N = Inf

## precomputations
G = interpolate_greens_function(greens_superlorentzian, 0, 100)
(R2sl, R2sl_dB1, R2sl_dB1_dT2s) = precompute_R2sl(100e-6, 1e-3, 5e-6, 15e-6, minimum(α), maximum(α), 0.7, 1.3)

(_R2sl, _R2sldT2s, _dR2sldB1) = R2sl_dB1_dT2s(TRF, ω1, B1, T2s)

## baseline IDE solution
m0  = [0.5 * (1 - m0s), 0, 0.5 * (1 - m0s), 0, m0s, 1]
m0g = [m0[1:5]; zeros(5); 1]
m = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl)) * m0


## derivative wrt. m0s
dm0s = 1e-6
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_m0s())) * m0g

md = exp(hamiltonian_linear(ω1, B1, ω0, TRF, (m0s + dm0s), R1, R2f, Rx, _R2sl)) * m0

mfd = (md - m) / dm0s
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. R1 
dR1 = 1e-6
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_R1())) * m0g

md = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, (R1 + dR1), R2f, Rx, _R2sl)) * m0

mfd = (md - m) / dR1
@test mg[6:10] ≈ mfd[1:5] rtol = max_error

## test derivative wrt. R2f
dR2f = 1e-6
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_R2f())) * m0g

md = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, (R2f + dR2f), Rx, _R2sl)) * m0

mfd = (md - m) / dR2f
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. Rx
dRx = 1e-5
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_Rx())) * m0g

md = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, (Rx + dRx), _R2sl)) * m0

mfd = (md - m) / dRx
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. T2s
dT2s = 1e-14
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_T2s())) * m0g

_dR2sl = R2sl(TRF, ω1, B1, (T2s+dT2s))
md = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _dR2sl)) * m0

mfd = (md - m) / dT2s
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. ω0
dω0 = 1e-6
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_ω0())) * m0g

md = exp(hamiltonian_linear(ω1, B1, (ω0 + dω0), TRF, m0s, R1, R2f, Rx, _R2sl)) * m0

mfd = (md - m) / dω0
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. B1
dB1 = 1e-6
mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1, R2f, Rx, _R2sl, _R2sldT2s, _dR2sldB1, grad_B1())) * m0g

_dR2sl = R2sl(TRF, ω1, (B1 + dB1), T2s)
md = exp(hamiltonian_linear(ω1, (B1 + dB1), ω0, TRF, m0s, R1, R2f, Rx, _dR2sl)) * m0

mfd = (md - m) / dB1
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## ########################################################################
# inversion pulse
###########################################################################
m = propagator_linear_inversion_pulse(ω1, TRF, B1, _R2sl, undef, undef, undef) * m0

## derivative wrt. m0s (generic for R1...)
dm0s = 1e-6
mg = propagator_linear_inversion_pulse(ω1, TRF, B1, _R2sl, _R2sldT2s, _dR2sldB1, grad_m0s()) * m0g

md = propagator_linear_inversion_pulse(ω1, TRF, B1, _R2sl, undef, undef, undef) * m0

mfd = (md - m) / dm0s
@test mg[6:10] ≈ mfd[1:5] rtol = max_error

## test derivative wrt. T2s
dT2s = 1e-14
mg = propagator_linear_inversion_pulse(ω1, TRF, B1, _R2sl, _R2sldT2s, _dR2sldB1, grad_T2s()) * m0g

_dR2sl = R2sl(TRF, ω1, B1, (T2s+dT2s))
md = propagator_linear_inversion_pulse(ω1, TRF, B1, _dR2sl, undef, undef, undef) * m0

mfd = (md - m) / dT2s
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. B1
dB1 = 1e-9
mg = propagator_linear_inversion_pulse(ω1, TRF, B1, _R2sl, _R2sldT2s, _dR2sldB1, grad_B1()) * m0g

_dR2sl = R2sl(TRF, ω1, (B1 + dB1), T2s)
md = propagator_linear_inversion_pulse(ω1, TRF, (B1 + dB1), _dR2sl, undef, undef, undef) * m0

mfd = (md - m) / dB1
@test mg[6:10] ≈ mfd[1:5] rtol = max_error
# norm(mg[6:10] - mfd[1:5]) / max(norm(mg[6:10]), norm(mfd[1:5]))