using BenchmarkTools
using MRIgeneralizedBloch
using Test
using FiniteDifferences

##
τmin = 0
τmax = 100

##
Gfull = greens_superlorentzian

print("Time to pre-calculate the interpolation:")
Ginte = @btime interpolate_greens_function(greens_superlorentzian, τmin, τmax)

##
for i = 1:100
    τ = τmin + (τmax - τmin) * rand()
    @test Gfull(τ) ≈ Ginte(τ)
end

## benchmark
τ = τmin + (τmax - τmin) * rand()
print("Time to evaluate the original     super-Lorenzian Green's function:")
@btime Gfull($τ)

print("Time to evaluate the interpolated super-Lorenzian Green's function:")
@btime Ginte($τ)

## #####################################################
# Test derivatives
########################################################
t = 100e-6
τ = 0
T2s = 10e-6

## Lorentzian
dGdT2s = dG_o_dT2s_x_T2s_lorentzian((t-τ)/T2s)/T2s
f_lorentzian = _T2s -> greens_lorentzian((t-τ)/_T2s) 
dGdT2s_fd = grad(central_fdm(5,1; factor=1e6,max_range=1e-8), f_lorentzian, T2s)[1]
@test dGdT2s ≈ dGdT2s_fd rtol = 1e-6

## Gaussian
dGdT2s = dG_o_dT2s_x_T2s_gaussian((t-τ)/T2s)/T2s
f_gaussian = _T2s -> greens_gaussian((t-τ)/_T2s) 
dGdT2s_fd = grad(central_fdm(5,1; factor=1e6,max_range=1e-8), f_gaussian, T2s)[1]
@test dGdT2s ≈ dGdT2s_fd rtol = 1e-6

## super-Lorentzian
dGdT2s = dG_o_dT2s_x_T2s_superlorentzian((t-τ)/T2s)/T2s
f_superlorentzian = _T2s -> greens_superlorentzian((t-τ)/_T2s) 
dGdT2s_fd = grad(central_fdm(5,1; factor=1e6,max_range=1e-8), f_superlorentzian, T2s)[1]
@test dGdT2s ≈ dGdT2s_fd rtol = 1e-6