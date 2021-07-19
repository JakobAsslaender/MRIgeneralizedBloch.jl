using MRIgeneralizedBloch
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native);

T2s = 10e-6 # s
t = 0 : 1e-6 : 1e-3
p = plot(1e3t, greens_lorentzian.(t ./ T2s), yaxis=:log, ylim=(1e-6,1), label="Lorentzian lineshape", xlabel="(t-τ) [ms]", ylabel="G((t-τ)/T2s)")
p = plot!(1e3t, greens_gaussian.(t ./ T2s), label="Gaussian lineshape")
p = plot!(1e3t, greens_superlorentzian.(t ./ T2s), label="super-Lorentzian l.")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

