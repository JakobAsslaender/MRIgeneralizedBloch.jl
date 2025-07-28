#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/NLLS.ipynb)

# # Non-Linear Least Square Fitting
# This section gives a brief overview of the interface to fit the generalized Bloch model to [hybrid-state free precession](https://www.nature.com/articles/s42005-019-0174-0) data. We use the [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl) package and supply the algorithm with analytic gradients that are calculated with the [`calculatesignal_linearapprox`](@ref) function that implements the linear approximation of the generalized Bloch model for a train of rectangular RF pulses.

# ## Basic Interface
# This tutorial uses the following packages:
using MRIgeneralizedBloch
using MAT
using LinearAlgebra
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# and we demonstrate the concept at the example of the RF pulse train that we published in the paper [Rapid quantitative magnetization transfer imaging: utilizing the hybrid state and the generalized Bloch model](https://arxiv.org/pdf/2207.08259.pdf):
control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))
α   = control["alpha"]
TRF = control["TRF"]
grad_moment = [:crusher; fill(:balanced, length(α)-1)]

TR = 3.5e-3
t = TR .* (1:length(TRF));

# As an example we can assume the following ground truth parameters
m0s = 0.15
R1f = 0.5 # 1/s
R2f = 17 # 1/s
Rex = 30 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
ω0 = 100 # rad/s
B1 = 0.9; # in units of B1_nominal

# precompute the linear approximation
R2slT = precompute_R2sl();

# and simulate the signal:
s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_moment)
s = vec(s)

# To make this example a bit more realistic, we add complex valued Gaussian noise:
s .+= 0.01 * randn(ComplexF64, size(s));

# Now we can fit the model to the noisy data:
qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, grad_moment)

# The last keyword argument is optional. It allows to recycle the precomputed `R2sl`, which improves speed. If not specified, it is re-calculated internally.

# The results are stored in a `struct` and we can access the fitted parameters with
qM.m0s
#-
qM.R1f # 1/s
#-
qM.R2f # 1/s
#-
qM.Rex # 1/s
#-
qM.R1s # 1/s
#-
1e6qM.T2s # μs
#-
qM.ω0 # rad/s
#-
qM.B1 # 1/B1_nominal

# We can also simulate the signal with the fitted parameters
s_fitted = calculatesignal_linearapprox(α, TRF, TR, qM.ω0, qM.B1, qM.m0s, qM.R1f, qM.R2f, qM.Rex, qM.R1s, qM.T2s, R2slT; grad_moment)
s_fitted = vec(s_fitted);

# and compare it to the noisy data:
p = plot(xlabel="t (s)", ylabel="signal (normalized)", legend=:topleft)
plot!(p, t, real.(s), label="Re(s)")
plot!(p, t, imag.(s), label="Im(s)")
plot!(p, t, real.(s_fitted), label="Re(s_fitted)")
plot!(p, t, imag.(s_fitted), label="Im(s_fitted)")
#md Main.HTMLPlot(p) #hide

# ## Bounds and Fixed Parameters
# Above example uses the default bounds

# `reM0 = (-Inf,   1,  Inf)`

# `imM0 = (-Inf,   0,  Inf)`

# `m0s  = (   0, 0.2,    1)`

# `R1f  = (   0, 0.3,  Inf)`

# `R2f  = (   0,  15,  Inf)`

# `Rex   = (   0,  20,  Inf)`

# `R1s  = (   0,   3,  Inf)`

# `T2s  = (8e-6,1e-5,12e-6)`

# `ω0   = (-Inf,   0,  Inf)`

# `B1   = (   0,   1,  1.5)`

# where the three entries refer to `(minimum, start_value, maximum)` (cf. [`fit_gBloch`](@ref)).

# With keyword arguments, one can modify each of these bounds. For example:
qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, m0s  = (0.1, 0.3, 0.5), grad_moment)

# starts the fit at `m0s = 0.3` and uses a lower bound of `0.1` and an upper bound of `0.5`. Alternatively, one also fix parameters to specified values:
qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, ω0 = 0, B1 = 1, grad_moment)

# In this case, the derivatives wrt. `ω0` and `B1` are not calculated and the result is accordingly
qM.ω0
#-
qM.B1

# ## Linear Compression
# As originally suggested by [McGivney et al.](https://ieeexplore.ieee.org/abstract/document/6851901) for MR Fingerprinting, the manifold of signal evolution or fingerprints is low rank and it is often beneficial to [reconstruct images directly in this domain](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26639). We can calculate a low rank basis with
sv = Array{ComplexF64}(undef, length(s), 50)
for i=1:size(sv,2)
    sv[:,i] = calculatesignal_linearapprox(α, TRF, TR, 500randn(), 0.8 + 0.4rand(), rand(), rand(), 20rand(), 30rand(), 3rand(), 8e-6+5e-6rand(), R2slT; grad_moment)
end
u, _, _ = svd(sv)
u = u[:,1:9];

# where the rank 9 was chosen heuristically. The noisy signal can be compressed with
sc = u' * s

# and fitted by calling [`fit_gBloch`](@ref) with the keyword argument `u=u`:
qM = fit_gBloch(sc, α, TRF, TR; R2slT=R2slT, u=u, grad_moment)


# ## Apparent ``R_1``
# Above fits tread `R1f` and `R1s` of the free and the semi-solid as independent parameters. As we discussed in our [paper](https://arxiv.org/pdf/2207.08259.pdf), many publications in the literature assume an apparent `R1a = R1f = R1s`. The corresponding model can be fitted by specifying `fit_apparentR1=true`:
R1a = 1 # 1/s
s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rex, R1a, T2s, R2slT; grad_moment)
qM = fit_gBloch(vec(s), α, TRF, TR; fit_apparentR1=true, R1a = (0, 0.7, Inf), R2slT=R2slT, grad_moment)

# Here, we specified the limits of `R1a` explicitly, which is optional.