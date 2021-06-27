"""
    greens_lorentzian(κ)

Evaluate the Green's function corresponding to a Lorentzian linshape at `κ = (t-τ)/T2s`.

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> greens_lorentzian((t-τ)/T2s)
4.5399929762484854e-5
```
"""
function greens_lorentzian(κ)
    exp(-κ)
end

"""
    dG_o_dT2s_x_T2s_lorentzian(κ)

Evaluate the derivative of Green's function, corresponding to a Lorentzian linshape, wrt. `T2s` at `κ = (t-τ)/T2s` and multiply it by `T2s`.

The multiplication is added so that the function merely depends on `κ = (t-τ)/T2s`. The actual derivative is given by `dG_o_dT2s_x_T2s_lorentzian((t-τ)/T2s)/T2s`.

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> dGdT2s = dG_o_dT2s_x_T2s_lorentzian((t-τ)/T2s)/T2s
45.39992976248485
```
"""
function dG_o_dT2s_x_T2s_lorentzian(κ)
    greens_lorentzian(κ) * κ
end

"""
    greens_gaussian(κ)

Evaluate the Green's function corresponding to a Gaussian linshape at `κ = (t-τ)/T2s`.

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> greens_gaussian((t-τ)/T2s)
1.9287498479639178e-22
```
"""
function greens_gaussian(κ)
    exp(-κ^2 / 2)
end

"""
    dG_o_dT2s_x_T2s_gaussian(κ)

Evaluate the derivative of Green's function, corresponding to a Gaussian linshape, wrt. `T2s` at `κ = (t-τ)/T2s` and multiply it by `T2s`.

The multiplication is added so that the function merely depends on `κ = (t-τ)/T2s`. The actual derivative is given by `dG_o_dT2s_x_T2s_gaussian((t-τ)/T2s)/T2s`.

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> dGdT2s = dG_o_dT2s_x_T2s_gaussian((t-τ)/T2s)/T2s
1.9287498479639177e-15
```
"""
function dG_o_dT2s_x_T2s_gaussian(κ)
    greens_gaussian(κ) * κ^2
end

"""
    greens_superlorentzian(κ)

Evaluate the Green's function corresponding to a super-Lorentzian linshape at `κ = (t-τ)/T2s`.

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> greens_superlorentzian((t-τ)/T2s)
0.1471246868094442
```
"""
function greens_superlorentzian(κ)
    quadgk(ζ -> exp(- κ^2 * (3ζ^2 - 1)^2 / 8), 0, sqrt(1/3), 1, order = 100)[1]
end

"""
    dG_o_dT2s_x_T2s_superlorentzian(κ)

Evaluate the derivative of Green's function, corresponding to a super-Lorentzian linshape, wrt. `T2s` at `κ = (t-τ)/T2s` and multiply it by `T2s`.

The multiplication is added so that the function merely depends on `κ = (t-τ)/T2s`. The actual derivative is given by `dG_o_dT2s_x_T2s_superlorentzian((t-τ)/T2s)/T2s`.

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> dGdT2s = dG_o_dT2s_x_T2s_superlorentzian((t-τ)/T2s)/T2s
15253.095033670965
```
"""
function dG_o_dT2s_x_T2s_superlorentzian(κ)
    quadgk(ζ -> exp(-κ^2 * (3ζ^2 - 1)^2 / 8) * (κ^2 * (3ζ^2 - 1)^2 / 4), 0, sqrt(1/3), 1, order = 100)[1]
end


"""
    interpolate_greens_function(f, κmin, κmax)

Interpolate the Green's function f in the range between κmin and κmax.

The interpolation uses the ApproxFun.jl package that incorporates Chebyshev polynomials and ensures an approximation to machine precision. 

# Examples
```jldoctest
julia> t = 100e-6;

julia> τ = 0;

julia> T2s = 10e-6;

julia> greens_superlorentzian((t-τ)/T2s)
0.1471246868094442

julia> Gint = interpolate_greens_function(greens_superlorentzian, 0, 20);


julia> Gint((t-τ)/T2s)
0.14712468680944407
```
"""
function interpolate_greens_function(f, κmin, κmax)
    Fun(f, κmin..κmax)
    # x = κmin : 0.001 : κmax
    # return CubicSplineInterpolation(x, f.(x))
end