using MRIgeneralizedBloch
using MAT
using LinearAlgebra
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))
α   = control["alpha"]
TRF = control["TRF"]

TR = 3.5e-3
t = TR .* (1:length(TRF));

m0s = 0.15
R1f = 0.5 # 1/s
R2f = 17 # 1/s
Rx = 30 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
ω0 = 100 # rad/s
B1 = 0.9; # in units of B1_nominal

R2slT = precompute_R2sl();

s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)
s = vec(s)

s .+= 0.01 * randn(ComplexF64, size(s));

qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT)

qM.m0s

qM.R1f # 1/s

qM.R2f # 1/s

qM.Rx # 1/s

qM.R1s # 1/s

1e6qM.T2s # μs

qM.ω0 # rad/s

qM.B1 # 1/B1_nominal

s_fitted = calculatesignal_linearapprox(α, TRF, TR, qM.ω0, qM.B1, qM.m0s, qM.R1f, qM.R2f, qM.Rx, qM.R1s, qM.T2s, R2slT)
s_fitted = vec(s_fitted);

p = plot(xlabel="t (s)", ylabel="signal (normalized)", legend=:topleft)
plot!(p, t, real.(s), label="Re(s)")
plot!(p, t, imag.(s), label="Im(s)")
plot!(p, t, real.(s_fitted), label="Re(s_fitted)")
plot!(p, t, imag.(s_fitted), label="Im(s_fitted)")

qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, m0s  = (0.1, 0.3, 0.5))

qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, ω0 = 0, B1 = 1)

qM.ω0

qM.B1

sv = Array{ComplexF64}(undef, length(s), 50)
for i=1:size(sv,2)
    sv[:,i] = calculatesignal_linearapprox(α, TRF, TR, 500randn(), 0.8 + 0.4rand(), rand(), rand(), 20rand(), 30rand(), 3rand(), 8e-6+5e-6rand(), R2slT)
end
u, _, _ = svd(sv)
u = u[:,1:9];

sc = u' * s

qM = fit_gBloch(sc, α, TRF, TR; R2slT=R2slT, u=u)

R1a = 1 # 1/s
s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rx, R1a, T2s, R2slT)
qM = fit_gBloch(vec(s), α, TRF, TR; fit_apparentR1=true, R1a = (0, 0.7, Inf), R2slT=R2slT)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

