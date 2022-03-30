#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/NLLS.ipynb)

# # Non-Linear Least Square Fitting
# This section gives a brief overview of the interface to fit the generalized Bloch model to [hybrid-state free precession](https://www.nature.com/articles/s42005-019-0174-0) data. We use the [LsqFit.jl](https://github.com/JuliaNLSolvers/LsqFit.jl) package and supply the algorithm with analytic gradients that are calcualted with the [`calculatesignal_linearapprox`](@ref) function that implements the linear approximation of the generalized Bloch model for a train of rectangular RF pulses.

# ## Basic Interface
# This tutorial uses the following packages:
using MRIgeneralizedBloch
using MAT
using LinearAlgebra
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# and we demonstrate the concept at the example of the RF pulse train that we publised in the paper [Rapid quantitative magnetization transfer imaging: utilizing the hybrid state and the generalized Bloch model](http://TODO.org):
control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))
α   = control["alpha"]
TRF = control["TRF"]

TR = 3.5e-3
t = TR .* (1:length(TRF));

# As an example we can assume the following ground truth parameters
m0s = 0.15
R1f = 0.5 # 1/s
R2f = 17 # 1/s
Rx = 30 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
ω0 = 100 # rad/s
B1 = 0.9; # in units of B1_nominal

# precomupte the linear approximation
R2slT = precompute_R2sl();

# and simulate the signal:
s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)
s = vec(s)

# To make this example a bit more realistic, we add complex valued Gaussian noise:
s .+= 0.01 * randn(ComplexF64, size(s));

# Now we can fit the model to the noisy data:
qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT)

# The last keyword argument is optional. It allows to recycle the precomputed `R2sl`, which improves speed. If not specified, it is re-calculated internally.

# We can access the fitted parameters with
qM.m0s
#-
qM.R1f # 1/s
#-
qM.R2f # 1/s
#-
qM.Rx # 1/s
#-
qM.R1s # 1/s
#-
1e6qM.T2s # μs
#-
qM.ω0 # rad/s
#-
qM.B1 # 1/B1_nominal

# We can also simulate the signal wiht the fitted parameters
s_fitted = calculatesignal_linearapprox(α, TRF, TR, qM.ω0, qM.B1, qM.m0s, qM.R1f, qM.R2f, qM.Rx, qM.R1s, qM.T2s, R2slT)
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

# `Rx   = (   0,  20,  Inf)`

# `R1s  = (   0,   3,  Inf)`

# `T2s  = (8e-6,1e-5,12e-6)`

# `ω0   = (-Inf,   0,  Inf)`

# `B1   = (   0,   1,  1.5)`

# where the three entries refer to `(minimum, start_value, maximum)`

# With keywords, one can modify each of these bounds. For example:
qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, m0s  = (0.1, 0.3, 0.5));

# starts the fit at `m0s = 0.3` and uses a lower bound of `0.1` and an upper bound of `0.5`. Alternatively, one also fix parameters to specified values:
qM = fit_gBloch(s, α, TRF, TR; R2slT=R2slT, ω0 = 0, B1 = 1);

# In this case, the graients wrt. `ω0` and `B1` are not calculated and the result is accordingly
qM.ω0
#-
qM.B1

# ## Linear Compression
# As originally suggested by [McGivney et al.](https://ieeexplore.ieee.org/abstract/document/6851901) for MR Fingerprinting, the manifold of signal evolution or fingerprints is low rank and it is often beneficial to [reconstruct images directly in this domain](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26639). We can calculate the corresponding basis functions with
sv = Array{ComplexF64}(undef, length(s), 50)
for i=1:size(sv,2)
    sv[:,i] = calculatesignal_linearapprox(α, TRF, TR, 500randn(), 0.8 + 0.4rand(), rand(), rand(), 20rand(), 30rand(), 3rand(), 8e-6+5e-6rand(), R2slT)
end
u, _, _ = svd(sv)
u = u[:,1:9];

# where the rank 9 was chosen heuristically. We can compress the noisy signal with
sc = u' * s

# and fit it by calling `fit_gBloch` with the keyword argument `u=u`:
qM = fit_gBloch(sc, α, TRF, TR; R2slT=R2slT, u=u)


# ## Apparent R1
# Above fits tread `R1f` and `R1s` of the free and the semi-solid as independant parameters. As we discussed in our [paper](http://TODO.org), many publications in the literature assume an apparent `R1a = R1f = R1s`. The corresponding model can be fitted by specifying `fit_apparentR1=true`:
R1a = 1 # 1/s
s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rx, R1a, T2s, R2slT)
qM = fit_gBloch(vec(s), α, TRF, TR; fit_apparentR1=true, R1a = (0, 0.7, Inf), R2slT=R2slT)

# Here, we specified the limits of `R1a` explicitely, which is optional.