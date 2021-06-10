using BenchmarkTools
using MRIgeneralizedBloch
using Test

##
τmin = 0
τmax = 100

##
Gfull = Greens_superLorentzian
Ginte = interpolate_Greens_Function(Greens_superLorentzian, τmin, τmax)

##
for i = 1:100
    τ = τmin + (τmax - τmin) * rand()
    @test Gfull(τ) ≈ Ginte(τ)
end

## benchmark
τ = τmin + (τmax - τmin) * rand()
print("Time to evaluate the original     super-Lorenzian Green's function:")
@btime Gfull(τ)

print("Time to evaluate the interpolated super-Lorenzian Green's function:")
@btime Ginte(τ)