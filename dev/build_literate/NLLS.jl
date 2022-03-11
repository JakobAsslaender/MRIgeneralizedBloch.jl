using MRIgeneralizedBloch
using MAT
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

# set parameters
ω0 = 0
B1 = 1
m0s = 0.15
R1f = 0.5
R2f = 1 / 65e-3
R1s = 3
T2s = 10e-6
Rx = 30
TR = 3.5e-3

control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))["control"]
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF
t = TR .* (1:length(TRF))

s_gBloch = vec(calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s))
s_Graham = vec(calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s))
s_linapp = vec(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT))

p = plot(TR * 1:length(TRF), real(s_gBloch), label="gBloch re(s)", legend=:topleft)
plot!(p, TR * 1:length(TRF), imag(s_gBloch), label="gBloch im(s)", legend=:topleft)
plot!(p, TR * 1:length(TRF), real(s_linapp), label="lin. approx. re(s)", legend=:topleft)
plot!(p, TR * 1:length(TRF), imag(s_linapp), label="lin. approx. im(s)", legend=:topleft)
plot!(p, TR * 1:length(TRF), real(s_Graham), label="Graham re(s)", legend=:topleft)
plot!(p, TR * 1:length(TRF), imag(s_Graham), label="Graham im(s)", legend=:topleft)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

