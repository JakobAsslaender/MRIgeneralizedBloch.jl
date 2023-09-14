using MRIgeneralizedBloch
using LinearAlgebra
BLAS.set_num_threads(1) # single threaded is faster in this case
using Optim             # provides the optimization algorithm
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

m0s = 0.15
R1f = 0.5   # 1/s
R2f = 15    # 1/s
Rx = 30     # 1/s
R1s = 3     # 1/s
T2s = 10e-6 # s
ω0 = 0      # rad/s
B1 = 1;     # in units of B1_nominal

Npulses = 200;

TR = 3.5e-3; # s

Npulses * TR

R2slT = precompute_R2sl();

grad_list = [grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()];

weights = transpose([0, 1, 0, 0, 0, 0, 0, 0, 0]);

α = abs.(sin.((1:Npulses) * 2π/Npulses));

TRF = 300e-6 .* one.(α);

α[1] = π
TRF[1] = 500e-6
isInversionPulse = [true; falses(length(α)-1)];

ω1 = α ./ TRF;

p1 = plot(TR*(1:Npulses), α ./ π, ylabel="α/π")
p2 = plot(TR*(1:Npulses), 1e6TRF, ylim=(0, 1e3), xlabel="t (s)", ylabel="TRF (μs)")
p = plot(p1, p2, layout=(2, 1), legend=:none)

(CRBm0s, grad_ω1, grad_TRF) = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, isInversionPulse=isInversionPulse)
CRBm0s

p1 = plot(TR*(1:Npulses), grad_ω1  .* ((-1) .^ (1:Npulses)), ylabel="∂CRB(m0s) / ∂ω1 (s/rad)")
p2 = plot(TR*(1:Npulses), grad_TRF .* ((-1) .^ (1:Npulses)), ylabel="∂CRB(m0s) / ∂TRF (1/s)", xlabel="t (s)")
p = plot(p1, p2, layout=(2, 1), legend=:none)

ω1_min  = 0      # rad/s
ω1_max  = 2e3π   # rad/s
TRF_min = 100e-6 # s
TRF_max = 500e-6; # s

x0 = MRIgeneralizedBloch.bound_ω1_TRF!(ω1, TRF; ω1_min = ω1_min, ω1_max = ω1_max, TRF_min = TRF_min, TRF_max = TRF_max)

G = similar(x0);

function fg!(F, G, x)
    ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x; ω1_min = ω1_min, ω1_max = ω1_max, TRF_min = TRF_min, TRF_max = TRF_max)

    (F, grad_ω1, grad_TRF) = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, isInversionPulse=isInversionPulse)
    F = abs(F)

    F += MRIgeneralizedBloch.second_order_α!(grad_ω1, grad_TRF, ω1, TRF; idx=isInversionPulse, λ=1e4)
    F += MRIgeneralizedBloch.RF_power!(grad_ω1, grad_TRF, ω1, TRF; idx=isInversionPulse, λ=1e-3, Pmax=3e6, TR=TR)
    F += MRIgeneralizedBloch.TRF_TV!(grad_TRF, ω1, TRF; idx=isInversionPulse, λ=1e3)

    MRIgeneralizedBloch.apply_bounds_to_grad!(G, x, grad_ω1, grad_TRF; ω1_min = ω1_min, ω1_max = ω1_max, TRF_min = TRF_min, TRF_max = TRF_max)
    return F
end;

result = optimize(Optim.only_fg!(fg!), # cost function
    x0,                                # initialization
    BFGS(),                            # algorithm
    Optim.Options(
        iterations=10_000,             # larger number as we use a time limit
        time_limit=(15*60)             # in seconds
        )
    )

ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(result.minimizer; ω1_min = ω1_min, ω1_max = ω1_max, TRF_min = TRF_min, TRF_max = TRF_max)
α = ω1 .* TRF;

(CRBm0s, grad_ω1, grad_TRF) = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, isInversionPulse=isInversionPulse)
CRBm0s

p1 = plot(TR*(1:Npulses), α ./ π, ylabel="α/π")
p2 = plot(TR*(1:Npulses), 1e6TRF, ylim=(0, 1e3), xlabel="t (s)", ylabel="TRF (μs)")
p = plot(p1, p2, layout=(2, 1), legend=:none)

m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT; output=:realmagnetization)
m = vec(m)

xf = [m[i][1] for i=1:length(m)]
yf = [m[i][2] for i=1:length(m)]
zf = [m[i][3] for i=1:length(m)]
xs = [m[i][4] for i=1:length(m)]
zs = [m[i][5] for i=1:length(m)]

p = plot(xlabel="t (s)", ylabel="m (normalized)")
plot!(p, TR*(1:Npulses), xf ./(1-m0s), label="xᶠ")
plot!(p, TR*(1:Npulses), yf ./(1-m0s), label="yᶠ")
plot!(p, TR*(1:Npulses), zf ./(1-m0s), label="zᶠ")
plot!(p, TR*(1:Npulses), xs ./   m0s , label="xˢ")
plot!(p, TR*(1:Npulses), zs ./   m0s , label="zˢ")

p = plot(xf, zf, xlabel="xf", ylabel="zf", framestyle = :zerolines, legend=:none)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
