#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/OCT.ipynb)

# # Optimal Control
# This section provides a brief introduction to the package's interface for sequence optimization. We use the [Cramer-Rao bound](https://en.wikipedia.org/wiki/Cramér–Rao_bound) (CRB) to assess a sequence's performance and optimize the amplitudes (``ω_1``) and durations (``T_\text{RF}``) of RF-pulses to reduce the CRB, assuming a [Balanced Hybrid-State Free Precession Pulse Sequence](@ref). For computational efficiency, the derivatives of the CRB wrt. ``ω_1`` and ``T_\text{RF}`` are calculated with the adjoint state method common in the [optimal control literature](https://www.sciencedirect.com/science/article/pii/S1090780703001538).

# For this tutorial, we use the following packages:
using MRIgeneralizedBloch
using LinearAlgebra
BLAS.set_num_threads(1) # single threaded is faster in this case
using Optim             # provides the optimization algorithm
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# Here, we optimize the pulse sequence for a predefined set of parameters:
m0s = 0.15
R1f = 0.5   # 1/s
R2f = 15    # 1/s
Rex = 30    # 1/s
R1s = 3     # 1/s
T2s = 10e-6 # s
ω0 = 0      # rad/s
B1 = 1;     # in units of B1_nominal

# and we optimize
Npulses = 200;

# pulses, spaced
TR = 3.5e-3; # s

# apart. The cyle duration of
Npulses * TR
# seconds is shorter than the optimal duration, which is in the range of 4-10s. We here use a small `Npulses` to speed up the computations. The [Linear Approximation](@ref) of the generalized Bloch model is precomputed with
R2slT = precompute_R2sl();

# In the calculation of the CRB, we account for following gradients:
grad_list = (grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1());

# and we sum up the CRB of all parameters, weighted by the following vector:
weights = transpose([0, 1, 0, 0, 0, 0, 0, 0, 0]);
# Note that the vector `weights` has one more entry compared to the `grad_list` vector, as the first derivative is always wrt. ``M_0``, regardless of `grad_list`. Here, we only optimize for the CRB of ``m_0^s``, while accounting for a fit of all 9 model parameters.

# We take some initial guess for the pulse train:
α = abs.(sin.((1:Npulses) * 2π/Npulses));

# initialize with a constant `TRF = 300μs`:
TRF = fill(300e-6, length(α));

# and define the first RF pulse as a 500μs inversion pulse by modifying vectors accordingly and by defining that crushers are played out before and after the inversion pulse:
α[1] = π
TRF[1] = 500e-6
grad_moment = [i == 1 ? :crusher : :balanced for i ∈ eachindex(α)];

# We note that inversion pulses are not optimized by this toolbox. We calculate the initial ``ω_1``
ω1 = α ./ TRF;

# and plot the initial control:
p1 = plot(TR*(1:Npulses), α ./ π, ylabel="α/π")
p2 = plot(TR*(1:Npulses), 1e6TRF, ylim=(0, 1e3), xlabel="t (s)", ylabel="TRF (μs)")
p = plot(p1, p2, layout=(2, 1), legend=:none)
#md Main.HTMLPlot(p) #hide

# With above defined weights, the function [`MRIgeneralizedBloch.CRB_gradient_OCT`](@ref) returns the CRB
CRBm0s, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment)
CRBm0s

# along with the gradients:
p1 = plot(TR*(1:Npulses), grad_ω1  .* ((-1) .^ (1:Npulses)), ylabel="∂CRB(m0s) / ∂ω1 (s/rad)")
p2 = plot(TR*(1:Npulses), grad_TRF .* ((-1) .^ (1:Npulses)), ylabel="∂CRB(m0s) / ∂TRF (1/s)", xlabel="t (s)")
p = plot(p1, p2, layout=(2, 1), legend=:none)
#md Main.HTMLPlot(p) #hide

# Note that we remove the oscillating nature of the gradient for the display.

# In this example, we limit the control to the following bounds
ω1_min  = fill(0,      length(ω1))  # rad/s
ω1_max  = fill(2e3π,   length(ω1))  # rad/s
TRF_min = fill(100e-6, length(ω1))  # s
TRF_max = fill(500e-6, length(ω1)); # s

# and the function [`MRIgeneralizedBloch.bound_ω1_TRF!`](@ref) modifies `ω1` and `TRF` to comply with these bounds and returns a single vector in the range `[-Inf, Inf]` that relates to the bounded control by a `tanh` transformation:
x0 = MRIgeneralizedBloch.bound_ω1_TRF!(ω1, TRF; ω1_min, ω1_max, TRF_min, TRF_max)

# Further, we initialize a gradient of the same length:
G = similar(x0);

# and define the cost function:
function fg!(F, G, x)
    ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x; ω1_min, ω1_max, TRF_min, TRF_max)

    F, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment)
    F = abs(F)

    F += MRIgeneralizedBloch.second_order_α!(grad_ω1, grad_TRF, ω1, TRF; grad_moment, λ=1e4)
    F += MRIgeneralizedBloch.RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1e-3, Pmax=3e6, TR)
    F += MRIgeneralizedBloch.TRF_TV!(grad_TRF, TRF; grad_moment, λ=1e3)

    MRIgeneralizedBloch.apply_bounds_to_grad!(G, x, grad_ω1, grad_TRF; ω1_min, ω1_max, TRF_min, TRF_max)
    return F
end;

# We perform the optimization with the package [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/), which requires the cost function `fg!(F, G, x)` to take the cost, the gradient, and the control as input variables and to over-write the gradient in place. The cost function calculates the gradient of the CRB with above described optimal control code and we, further, add some regularization terms: [`MRIgeneralizedBloch.second_order_α!`](@ref) penalizes the curvature of α, which smoothes the flip angle train and helps ensuring the [hybrid state conditions](https://www.nature.com/articles/s42005-019-0174-0). The penalty [`MRIgeneralizedBloch.RF_power!`](@ref) penalizes the power deposition of the RF-pulse train if ``\Sigma_i(ω_1^2[i] ⋅ T_{\text{RF}}[i]) / T_{\text{cycle}} ≥ P_{\max}`` and helps with compliance to safety limits. Assuming a reasonable `λ`, the optimization will converge to an average RF-power deposition equal to or less than `Pmax` in units of (rad/s)². Heuristically, the value `Pmax=3e6` (rad/s)² proofed to be a reasonable choice for 3T systems. The penalty [`MRIgeneralizedBloch.TRF_TV!`](@ref) penalizes fast fluctuations of ``T_\text{RF}``. This penalty is justified by the knowledge that fluctuations of the control have negligible effect if they are fast compared to the biophysical time constants. We note, however, that this penalty is not required and rather ensure *beauty* of the result and speeds up convergence.

# With all this in place, we can start the actual optimization
result = optimize(Optim.only_fg!(fg!), # cost function
    x0,                                # initialization
    BFGS(),                            # algorithm
    Optim.Options(
        iterations=10_000,             # larger number as we use a time limit
        time_limit=(15*60),            # in seconds
        )
    )


# After transforming the optimized control back into the space of bounded ``ω_1`` and ``T_\text{RF}`` values
ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(result.minimizer; ω1_min, ω1_max, TRF_min, TRF_max)
α = ω1 .* TRF;

# we analyze the CRB(m0s):
CRBm0s, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment)
CRBm0s

# and observe a substantial reduction. Further, we plot the optimized control:
p1 = plot(TR*(1:Npulses), α ./ π, ylabel="α/π")
p2 = plot(TR*(1:Npulses), 1e6TRF, ylim=(0, 1e3), xlabel="t (s)", ylabel="TRF (μs)")
p = plot(p1, p2, layout=(2, 1), legend=:none)
#md Main.HTMLPlot(p) #hide

# To further analyze the results, we can calculate and plot all magnetization components:
m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; output=:realmagnetization)
m = vec(m)

xf = [m[i][1] for i ∈ eachindex(m)]
yf = [m[i][2] for i ∈ eachindex(m)]
zf = [m[i][3] for i ∈ eachindex(m)]
xs = [m[i][4] for i ∈ eachindex(m)]
zs = [m[i][5] for i ∈ eachindex(m)]

p = plot(xlabel="t (s)", ylabel="m (normalized)")
plot!(p, TR*(1:Npulses), xf ./(1-m0s), label="xᶠ")
plot!(p, TR*(1:Npulses), yf ./(1-m0s), label="yᶠ")
plot!(p, TR*(1:Npulses), zf ./(1-m0s), label="zᶠ")
plot!(p, TR*(1:Npulses), xs ./   m0s , label="xˢ")
plot!(p, TR*(1:Npulses), zs ./   m0s , label="zˢ")
#md Main.HTMLPlot(p) #hide

# And we can also plot the dynamics of the free spin pool on the Bloch sphere:
p = plot(xf, zf, xlabel="xf", ylabel="zf", framestyle = :zerolines, legend=:none)
#md Main.HTMLPlot(p) #hide

# As yᶠ is close to zero in this particular case, we neglect it in this 2D plot.