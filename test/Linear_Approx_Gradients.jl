using DifferentialEquations
using Test
using MRIgeneralizedBloch
using MRIgeneralizedBloch: hamiltonian_linear, propagator_linear_crushed_pulse
using FiniteDifferences

##
max_error = 1e-5

## set parameters
α  = π
ω1 = α / 500e-6
ω0 = 200.0
B1 = 1.0
m0s = 0.15
R1f = 0.3
R2f = 1 / 65e-3
R1s = 2.
T2s = 10e-6
K = 30.0
nTR = 0.07
TRF = 500e-6
N = Inf

## precomputations
(R2sl, dR2sldT2s, dR2sldB1) = precompute_R2sl()

_R2sl = R2sl(TRF, α, B1, T2s)
_dR2sldT2s = dR2sldT2s(TRF, α, B1, T2s)
_dR2sldB1 = dR2sldB1(TRF, α, B1, T2s)

## baseline IDE solution
m0  = [0.5 * (1 - m0s), 0, 0.5 * (1 - m0s), 0, m0s, 1]
m0g = [m0[1:5]; zeros(5); 1]
m = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl)) * m0


## derivative wrt. m0s
f_m0s = _m0s -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, _m0s, R1f, R2f, K, nTR, R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_m0s, m0s)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_m0s())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. R1f
f_R1f = _R1f -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, _R1f, R2f, K, nTR, R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_R1f, R1f)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_R1f())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. R2f
f_R2f = _R2f -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, _R2f, K, nTR, R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_R2f, R2f)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_R2f())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. K
f_K = _K -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, _K, nTR, R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_K, K)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_K())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. nTR
f_nTR = _nTR -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, _nTR, R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_nTR, nTR)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_nTR())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. R1s
f_R1s = _R1s -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, _R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_R1s, R1s)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_R1s())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. T2s
f_T2s = _T2s -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, R2sl(TRF,α,B1,_T2s))) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6,max_range=1e-8), f_T2s, T2s)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_T2s())) * m0g
@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. ω0
f_ω0 = _ω0 -> exp(hamiltonian_linear(ω1, B1, _ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_ω0, ω0)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_ω0())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. B1
f_B1 =_B1 -> exp(hamiltonian_linear(ω1, _B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, R2sl(TRF,α,_B1,T2s))) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6,max_range=1e-2), f_B1, B1)[1]


mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, K, nTR, R1s, _R2sl, _dR2sldT2s, _dR2sldB1, grad_B1())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. R1a
R1a = 1.
f_R1a = _R1a -> exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, _R1a, R2f, K, nTR, _R1a, _R2sl)) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6), f_R1a, R1a)[1]

mg = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1a, R2f, K, nTR, R1a, _R2sl, _dR2sldT2s, _dR2sldB1, grad_R1a())) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## ########################################################################
# inversion pulse
###########################################################################
## derivative wrt. m0s
mg = propagator_linear_crushed_pulse(ω1, TRF, B1, _R2sl, _dR2sldT2s, _dR2sldB1, grad_m0s()) * m0g

@test mg[6:10] ≈ zeros(5) atol = max_error


## test derivative wrt. T2s
f_inv_T2s = _T2s -> propagator_linear_crushed_pulse(ω1, TRF, B1, R2sl(TRF,α,B1,_T2s), undef, undef, undef) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6,max_range=1e-8), f_inv_T2s, T2s)[1]

mg = propagator_linear_crushed_pulse(ω1, TRF, B1, _R2sl, _dR2sldT2s, _dR2sldB1, grad_T2s()) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error


## test derivative wrt. B1
f_inv_B1 = _B1 -> propagator_linear_crushed_pulse(ω1, TRF, _B1, R2sl(TRF, α, (_B1), T2s), undef, undef, undef) * m0
mfd = jacobian(central_fdm(5,1; factor=1e6, max_range=1e-2), f_inv_B1, B1)[1]

mg = propagator_linear_crushed_pulse(ω1, TRF, B1, _R2sl, _dR2sldT2s, _dR2sldB1, grad_B1()) * m0g

@test mg[6:10] ≈ mfd[1:5] rtol = max_error