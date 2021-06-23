using MRIgeneralizedBloch
using MAT
using Plots
plotlyjs(ticks=:native)
theme(:lime)

## set parameters
ω0 = 0.0
B1 = 1.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TR = 3.5e-3

control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../examples/control_MT_v3p2_TR3p5ms_discretized.mat")))["control"]
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF

###############################################################
# calculate and plot magnetization vectors (xf,yf,zf,zs)
###############################################################
## IDE solution (no Gradients)
m_gBloch = calculatemagnetization_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], 2)
plot( TR * 1:length(TRF), m_gBloch[:,1] ./ (1 - m0s), label="xf / m0f", legend=:topleft)
plot!(TR * 1:length(TRF), m_gBloch[:,2] ./ (1 - m0s), label="yf / m0f")
plot!(TR * 1:length(TRF), m_gBloch[:,3] ./ (1 - m0s), label="zf / m0f")
plot!(TR * 1:length(TRF), m_gBloch[:,4] ./ m0s, label="zs / m0s")

## Linear approximation 
R2slT = precompute_R2sl(minimum(TRF), maximum(TRF), T2s, T2s, minimum(α), maximum(α), B1, B1)

m_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT; output=:realmagnetization)
m_linapp = [m_linapp[i][j] for i=1:size(m_linapp,1),j=1:5]

plot!(TR * 1:length(TRF), m_linapp[:,1] ./ (1 - m0s), label="L: xf / m0f", legend=:topleft)
plot!(TR * 1:length(TRF), m_linapp[:,2] ./ (1 - m0s), label="L: yf / m0f")
plot!(TR * 1:length(TRF), m_linapp[:,3] ./ (1 - m0s), label="L: zf / m0f")
plot!(TR * 1:length(TRF), m_linapp[:,5] ./ m0s, label="L: zs / m0s")


## Graham's solution
m_Graham = calculatemagnetization_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], 2)
plot!(TR * 1:length(TRF), m_Graham[:,1] ./ (1 - m0s), label="G: xf / m0f")
plot!(TR * 1:length(TRF), m_Graham[:,2] ./ (1 - m0s), label="G: yf / m0f")
plot!(TR * 1:length(TRF), m_Graham[:,3] ./ (1 - m0s), label="G: zf / m0f")
plot!(TR * 1:length(TRF), m_Graham[:,4] ./ m0s, label="G: zs / m0s")

###############################################################
# calculate and plot signal vectors (xf + 1im * yf)
###############################################################
##
s_gBloch = calculatesignal_gbloch_ide(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, 2)
s_Graham = calculatesignal_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, 2)
s_linapp = vec(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT))


plot( TR * 1:length(TRF), real(s_gBloch), label="gBloch re(s)", legend=:topleft)
plot!(TR * 1:length(TRF), imag(s_gBloch), label="gBloch im(s)", legend=:topleft)
plot!(TR * 1:length(TRF), real(s_linapp), label="lin. approx. re(s)", legend=:topleft)
plot!(TR * 1:length(TRF), imag(s_linapp), label="lin. approx. im(s)", legend=:topleft)
plot!(TR * 1:length(TRF), real(s_Graham), label="Graham re(s)", legend=:topleft)
plot!(TR * 1:length(TRF), imag(s_Graham), label="Graham im(s)", legend=:topleft)