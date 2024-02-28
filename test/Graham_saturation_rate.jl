##
using MRIgeneralizedBloch
using SpecialFunctions
using Test

## rectangular pulse
T2s = 10e-6
TRF = 5e-3
α = π
ω0 = 10e3 * 2π # rad/s
ω1 = α/TRF

Rrf_sf = graham_saturation_rate_single_frequency(ω0 -> lineshape_superlorentzian(ω0, T2s), ω1, TRF, ω0)
Rrf_sp = graham_saturation_rate_spectral(ω0 -> lineshape_superlorentzian(ω0, T2s), ω1, TRF, ω0)

@test Rrf_sf ≈ Rrf_sp rtol = 1e-2

## shaped pulse
shape_param = 16 # what is the right choice?
f_ω1 = (t) -> exp(- (t-TRF/2)^2 / TRF^2 * shape_param) / (sqrt(π/shape_param) * TRF * erf(sqrt(shape_param)/2)) * α

Rrf_sf = graham_saturation_rate_single_frequency(ω0 -> lineshape_superlorentzian(ω0, T2s), f_ω1, TRF, ω0)
Rrf_sp = graham_saturation_rate_spectral(ω0 -> lineshape_superlorentzian(ω0, T2s), f_ω1, TRF, ω0)

@test Rrf_sf ≈ Rrf_sp rtol = 1e-2