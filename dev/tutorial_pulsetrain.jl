#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/tutorial_pulsetrain.ipynb)

# # Balanced Hybrid-State Free Precession Pulse Sequence
# This section explains the interface for calculating the spin evolution during a train of RF pulses, assuming balanced gradient moments. For this, we need the following packages:

using MRIgeneralizedBloch
using MAT
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# and we use the pulse train described in the paper [Rapid quantitative magnetization transfer imaging: utilizing the hybrid state and the generalized Bloch model](http://TODO.org):
control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))
α   = control["alpha"]
TRF = control["TRF"]

# and setting TR...
TR = 3.5e-3
t = TR .* (1:length(TRF))

p1 = plot(t, α/π, ylabel="α/π", label=:none)
p2 = plot(t, TRF, xlabel="t (s)", ylabel="TRF (s)", label=:none)
p = plot(p1, p2, layout=(2,1))
#md Main.HTMLPlot(p) #hide


# We simulate the signal for the following biophysical parameters:
m0s = 0.15
R1f = 0.5 # 1/s
R2f = 15 # 1/s
Rx = 30 # 1/s
R1s = 3 # 1/s
T2s = 10e-6 # s
ω0 = 0 # rad/s
B1 = 1; # in units of B1_nominal

# For speed purposes, it is advisable to use the linear approximation of the generalized Bloch model, which requires a precomputed ``R_2^{s,l}``
R2slT = precompute_R2sl();

# Now we have everything set up to calculate the signal. By default, the output is a complex valued array where each element describes the transversal magnetization ``x^f + i y^f`` of the free spin pool in each ``T_\text{R}``:
s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

p = plot(xlabel="t (s)", ylabel="signal (normalized)"; legend=:topleft)
plot!(p, t, real.(vec(s_linapp)), label="Re(s); lin. approx.")
plot!(p, t, imag.(vec(s_linapp)), label="Im(s); lin. approx.")
#md Main.HTMLPlot(p) #hide

# For comparison, we can also solve the full integro-differential equation (IDE) for each RF pulse, which is more accurate, but much slower:
s_ide = calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s)

plot!(p, t, real.(vec(s_ide)), label="Re(s); IDE")
plot!(p, t, imag.(vec(s_ide)), label="Im(s); IDE")
#md Main.HTMLPlot(p) #hide

# Clicking on the legend entries allows to select and de-select individual graphs.


# ## Real-valued magnetization vector
# As an alternative to the complex-valued signal, we can also calculate the full mangetization vector ``(x^f, y^f, z^f, x^s, z^s, 1)`` by supplying the keyword argument `output=:realmagnetization`. Here, ``x``, ``y``, ``z`` denote the dimensions in space, the superscripts ``f`` and ``s`` denote the free and the semi-solid spin pool, respectively. We neglect the ``y^s`` component, assuming (without loss of generality) ωₓ = 0 and given that ``R_2^{s,l} \gg ω_0``.

m_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT;
    output=:realmagnetization)

p = plot(xlabel="t (s)", ylabel="magnetization (normalized)"; legend=:topleft)
plot!(p, t, [m_linapp[i][1] for i=1:size(m_linapp,1)] ./ (1 - m0s), label="xᶠ / m₀ᶠ")
plot!(p, t, [m_linapp[i][2] for i=1:size(m_linapp,1)] ./ (1 - m0s), label="yᶠ / m₀ᶠ")
plot!(p, t, [m_linapp[i][3] for i=1:size(m_linapp,1)] ./ (1 - m0s), label="zᶠ / m₀ᶠ")
plot!(p, t, [m_linapp[i][4] for i=1:size(m_linapp,1)] ./      m0s , label="xˢ / m₀ˢ")
plot!(p, t, [m_linapp[i][5] for i=1:size(m_linapp,1)] ./      m0s , label="zˢ / m₀ˢ")
#md Main.HTMLPlot(p) #hide

# ## Gradients
# The same interface can also be used to calculate the derivatives of the signal wrt. the biophysical parameters. One can specify any subset of derivatives in any order with a vector of identifyer objects:
grad_list=[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()];

# Calling the function `calculatesignal_linearapprox` with the keyword argument `grad_list` and this vector
s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT;
    grad_list=grad_list);

# returns the derivatives in the specified order:
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
#md Main.HTMLPlot(p) #hide

# Note that the first row is always the signal itself, which is equivalent to ∂s/∂M₀, as this toolbox always assumens M₀ = 1.

# ### Apparent R₁
# Above code calculates separate derivatives for ``R_1^f`` and ``R_1^s``. Yet, many publications, including our own paper ["Rapid quantitative magnetization transfer imaging: utilizing the hybrid state and the generalized Bloch model"](http://TODO.org) assumes an apparent longitudinal relaxation rate ``R_1^a = R_1^f = R_1^f``. The derivatives wrt. this apparent relaxation rate can be calcuated with

R1a = 1 # 1/s
grad_list=[grad_R1a()]
s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rx, R1a, T2s, R2slT;
    grad_list=grad_list)

p = plot(xlabel="t (s)", ylabel="signal (normalized)"; legend=:topleft)
plot!(p, t, real.(s_linapp[:,1,1]       ), label="Re(∂s/∂M₀)/M₀")
plot!(p, t, real.(s_linapp[:,1,2] .* R1a), label="Re(∂s/∂R₁ᵃ)*R₁ᵃ")
#md Main.HTMLPlot(p) #hide

# Note that `R1a` appears here twice in the arguments of the `calculatesignal_linearapprox` in place of `R1f` and `R1s`.