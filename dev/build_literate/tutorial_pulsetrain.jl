using MRIgeneralizedBloch
using MAT
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))
α   = control["alpha"]
TRF = control["TRF"]

TR = 3.5e-3
t = TR .* (1:length(TRF))

p1 = plot(t, α/π, ylabel="α/π", label=:none)
p2 = plot(t, TRF, xlabel="t (s)", ylabel="TRF (s)", label=:none)
p = plot(p1, p2, layout=(2,1))

m0s = 0.15
R1f = 0.5 # 1/s
R2f = 15 # 1/s
Rx = 30 # 1/s
R1s = 3 # 1/s
T2s = 10e-6 # s
ω0 = 0 # rad/s
B1 = 1; # in units of B1_nominal

R2slT = precompute_R2sl();

s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

p = plot(xlabel="t (s)", ylabel="signal (normalized)"; legend=:topleft)
plot!(p, t, real.(vec(s_linapp)), label="Re(s); lin. approx.")
plot!(p, t, imag.(vec(s_linapp)), label="Im(s); lin. approx.")

s_ide = calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s)

plot!(p, t, real.(vec(s_ide)), label="Re(s); IDE", legend=:topleft)
plot!(p, t, imag.(vec(s_ide)), label="Im(s); IDE", legend=:topleft)

m_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT;
    output=:realmagnetization)

p = plot(xlabel="t (s)", ylabel="magnetization (normalized)"; legend=:topleft)
plot!(p, t, [m_linapp[i][1] for i=1:size(m_linapp,1)] ./ (1 - m0s), label="xᶠ / m₀ᶠ")
plot!(p, t, [m_linapp[i][2] for i=1:size(m_linapp,1)] ./ (1 - m0s), label="yᶠ / m₀ᶠ")
plot!(p, t, [m_linapp[i][3] for i=1:size(m_linapp,1)] ./ (1 - m0s), label="zᶠ / m₀ᶠ")
plot!(p, t, [m_linapp[i][4] for i=1:size(m_linapp,1)] ./      m0s , label="xˢ / m₀ˢ")
plot!(p, t, [m_linapp[i][5] for i=1:size(m_linapp,1)] ./      m0s , label="zˢ / m₀ˢ")

grad_list=[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()];

s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT;
    grad_list=grad_list);

p = plot(xlabel="t (s)", ylabel="signal (normalized)"; legend=:topleft)
plot!(p, t, real.(s_linapp[:,1,1]       ), label="Re(∂s/∂M₀ )*M₀")
plot!(p, t, real.(s_linapp[:,1,2] .* m0s), label="Re(∂s/∂m₀ˢ)*m₀ˢ")
plot!(p, t, real.(s_linapp[:,1,3] .* R1f), label="Re(∂s/∂R₁ᶠ)*R₁ᶠ")
plot!(p, t, real.(s_linapp[:,1,4] .* R2f), label="Re(∂s/∂R₂ᶠ)*R₂ᶠ")
plot!(p, t, real.(s_linapp[:,1,5] .* Rx ), label="Re(∂s/∂Rₓ )*Rₓ ")
plot!(p, t, real.(s_linapp[:,1,6] .* R1s), label="Re(∂s/∂R₁ˢ)*R₁ˢ")
plot!(p, t, real.(s_linapp[:,1,7] .* T2s), label="Re(∂s/∂T₂ˢ)*T₂ˢ")
plot!(p, t, real.(s_linapp[:,1,8] .* ω0 ), label="Re(∂s/∂ω₀ )*ω₀ ")
plot!(p, t, real.(s_linapp[:,1,9] .* B1 ), label="Re(∂s/∂B₁ )*B₁ ")

R1a = 1 # 1/s
grad_list=[grad_R1a()]
s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rx, R1a, T2s, R2slT; grad_list=grad_list)

p = plot(xlabel="t (s)", ylabel="signal (normalized)"; legend=:topleft)
plot!(p, t, real.(s_linapp[:,1,1]       ), label="Re(∂s/∂M₀)/M₀")
plot!(p, t, real.(s_linapp[:,1,2] .* R1a), label="Re(∂s/∂R₁ᵃ)*R₁ᵃ")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

