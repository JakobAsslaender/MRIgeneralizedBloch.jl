"""
    precompute_R2sl([;TRF_min=100e-6, TRF_max=500e-6, T2s_min=5e-6, T2s_max=21e-6, ω1_max=π/TRF_max, B1_max=1.5, greens=greens_superlorentzian])

Pre-compute and interpolate the linearized `R2sl(TRF, α, B1, T2s)` and its derivatives `dR2sldB1(TRF, α, B1, T2s)`, `R2sldT2s(TRF, α, B1, T2s)` etc. in the range specified by the arguments.

The function solves the generalized Bloch equations of an isolated semi-solid pool for values in the specified range, calulates the linearized R2sl that minimizes the error of `zs` at the end of the RF-pulse, and interpolates between the different samples.

# Optional Arguments:
- `TRF_min::Number`: lower bound of the RF-pulse duration range in seconds
- `TRF_max::Number`: upper bound of the RF-pulse duration range in seconds
- `T2s_min::Number`: lower bound of the `T2s` range in seconds
- `T2s_max::Number`: upper bound of the `T2s` range in seconds
- `ω1_max::Number`: upper bound of the Rabi frequency ω1, the default is the frequency of a 500μs long π-pulse
- `B1_max::Number`: upper bound of the B1 range, normalized so that `B1 = 1` corresponds to a perfectly calibrated RF field
- `greens=greens_superlorentzian`: Greens function in the form `G(κ) = G((t-τ)/T2s)`. This package supplies the three Greens functions `greens=greens_superlorentzian` (default), `greens=greens_lorentzian`, and `greens=greens_gaussian`

# Examples
```jldoctest
julia> R2slT = precompute_R2sl();

julia> R2sl, dR2sldB1, R2sldT2s, _ = precompute_R2sl(TRF_min=100e-6, TRF_max=500e-6, T2s_min=5e-6, T2s_max=15e-6, ω1_max=π/500e-6, B1_max=1.3, greens=greens_gaussian);

```
"""
function precompute_R2sl(;TRF_min=100e-6, TRF_max=500e-6, T2s_min=5e-6, T2s_max=21e-6, ω1_max=π/TRF_max, B1_max=1.5, greens=greens_superlorentzian)

    # interpolate super-Lorentzian Green's function for speed purposes
    if greens == greens_superlorentzian
        G = interpolate_greens_function(greens, 0, TRF_max / T2s_min)
    else
        G = greens
    end

    function calculate_z(τ, Ω, G)
        mfun(p, t) = [1.0]

        function hamiltonian_1D!(du, u, mfun, p::NTuple{3,Any}, t)
            ω1, T2s, g = p
            du[1] = -ω1^2 * quadgk(x -> g((t - x) / T2s) * mfun(p, x)[1], 0.0, t)[1]
        end

        z = solve(DDEProblem(hamiltonian_1D!, [1.0], mfun, (0, τ), (Ω, 1, G)), MethodOfSteps(DP8()))
        return z
    end

    function calculate_R2sl(z_fun, τ, Ω)
        z = z_fun(τ)[1]
        function f!(F, ρv)
            ρ = ρv[1]
            s = sqrt(Complex(ρ^2 - (2Ω)^2))
            x = exp(-ρ * τ / 2) * (cosh(τ * s / 2) + ρ / s * sinh(τ * s / 2))
            F[1] = real(x) - z
        end
        function j!(J, ρv)
            ρ = ρv[1]
            s = sqrt(Complex(ρ^2 - (2Ω)^2))
            x = 2 * Ω^2 * exp(-ρ * τ / 2) * (τ * s * cosh(τ * s / 2) - 2sinh(τ * s / 2)) / s^3
            J[1] = real(x)
        end

        sol = nlsolve(f!, j!, [0.1])
        return sol.zero[1]
    end

    τv = range(TRF_min / T2s_max, TRF_max / T2s_min; length=2^10)
    Ωv = range(0, B1_max * ω1_max * T2s_max; length=2^6)

    A = Matrix{Float64}(undef, length(τv), length(Ωv))
    @batch minbatch=8 for iΩ ∈ 2:length(Ωv)
        τmax = min(τv[end], TRF_max * ω1_max / Ωv[iΩ])
        z_fun = calculate_z(τmax, Ωv[iΩ], G)
        for iτ in eachindex(τv)
            τ = min(τv[iτ], TRF_max * ω1_max / Ωv[iΩ])
            A[iτ,iΩ] = calculate_R2sl(z_fun, τ, Ωv[iΩ])
        end
    end
    A[:,1] .= A[:,2] # extrapolation hack as the fit does not work with Ω = 0

    f = CubicSplineInterpolation((τv, Ωv), A)
    dfdτ(   τ, Ω) = Interpolations.gradient(f, τ, Ω)[1]
    dfdΩ(   τ, Ω) = Interpolations.gradient(f, τ, Ω)[2]
    d2fdτ2( τ, Ω) = Interpolations.hessian( f, τ, Ω)[1,1]
    d2fdΩ2( τ, Ω) = Interpolations.hessian( f, τ, Ω)[2,2]
    d2fdτdΩ(τ, Ω) = Interpolations.hessian( f, τ, Ω)[2,1]

    R2sl(TRF, α, B1, T2s) = f(TRF / T2s, B1 * α * T2s / TRF) / T2s

    dR2sldT2s(TRF, α, B1, T2s) = -dfdτ(TRF / T2s, B1 * α * T2s / TRF) * TRF / T2s^3 - f(TRF / T2s, B1 * α * T2s / TRF) / T2s^2 + dfdΩ(TRF / T2s, B1 * α * T2s / TRF) * B1 * α / TRF / T2s
    dR2sldB1(TRF, α, B1, T2s)  = dfdΩ(TRF / T2s, B1 * α * T2s / TRF) * α / TRF

    dR2sldω1(TRF, α, B1, T2s)  = dfdΩ(TRF / T2s, B1 * α * T2s / TRF) * B1
    dR2sldTRF(TRF, α, B1, T2s) = dfdτ(TRF / T2s, B1 * α * T2s / TRF) / T2s^2


    dR2sldT2sdω1(TRF, α, B1, T2s) = -d2fdτdΩ(TRF / T2s, B1 * α * T2s / TRF) * B1 * TRF / T2s^2 + d2fdΩ2(TRF / T2s, B1 * α * T2s / TRF) * B1^2 * α / TRF


    dR2sldB1dω1(TRF, α, B1, T2s) = dfdΩ(TRF / T2s, B1 * α * T2s / TRF) + d2fdΩ2(TRF / T2s, B1 * α * T2s / TRF) * B1 * α * T2s  / TRF
    dR2sldT2sTRF(TRF, α, B1, T2s) = -dfdτ(TRF / T2s, B1 * α * T2s / TRF) / T2s^3 - d2fdτ2(TRF / T2s, B1 * α * T2s / TRF) * TRF / T2s^4 - dfdτ(TRF / T2s, B1 * α * T2s / TRF) / T2s^3 + d2fdτdΩ(TRF / T2s, B1 * α * T2s / TRF) * B1 * α / TRF / T2s^2
    dR2sldB1dTRF(TRF, α, B1, T2s) = d2fdτdΩ(TRF / T2s, B1 * α * T2s / TRF) * α / TRF / T2s

    return (R2sl, dR2sldT2s, dR2sldB1, dR2sldω1, dR2sldTRF, dR2sldT2sdω1, dR2sldB1dω1, dR2sldT2sTRF, dR2sldB1dTRF)
end


function evaluate_R2sl_vector(α, TRF, B1, T2s, R2slT, grad_list)
    _R2s = similar(α)
    _dR2sdT2s = similar(α)
    _dR2sdB1 = similar(α)

    for i = 1:length(α)
        _R2s[i] = R2slT[1](TRF[i], α[i], B1, T2s)
    end
    if any(isa.(grad_list, grad_T2s))
        for i = 1:length(α)
            _dR2sdT2s[i] = R2slT[2](TRF[i], α[i], B1, T2s)
        end
    end
    if any(isa.(grad_list, grad_B1))
        for i = 1:length(α)
            _dR2sdB1[i] = R2slT[3](TRF[i], α[i], B1, T2s)
        end
    end
    return (_R2s, _dR2sdT2s, _dR2sdB1)
end

function evaluate_R2sl_vector_OCT(α, TRF, B1, T2s, R2slT, grad_list)
    (_R2s, _dR2sdT2s, _dR2sdB1) = evaluate_R2sl_vector(α, TRF, B1, T2s, R2slT, grad_list)

    _dR2sldω1      = similar(α)
    _dR2sldTRF     = similar(α)
    _dR2sldT2sdω1  = similar(α)
    _dR2sldB1dω1   = similar(α)
    _dR2sldT2sdTRF = similar(α)
    _dR2sldB1dTRF  = similar(α)

    for i = 1:length(α)
        _dR2sldω1[i]      = R2slT[4](TRF[i], α[i], B1, T2s)
        _dR2sldTRF[i]     = R2slT[5](TRF[i], α[i], B1, T2s)
        _dR2sldT2sdω1[i]  = R2slT[6](TRF[i], α[i], B1, T2s)
        _dR2sldB1dω1[i]   = R2slT[7](TRF[i], α[i], B1, T2s)
        _dR2sldT2sdTRF[i] = R2slT[8](TRF[i], α[i], B1, T2s)
        _dR2sldB1dTRF[i]  = R2slT[9](TRF[i], α[i], B1, T2s)
    end
    return (_R2s, _dR2sdT2s, _dR2sdB1, _dR2sldω1, _dR2sldTRF, _dR2sldT2sdω1, _dR2sldB1dω1, _dR2sldT2sdTRF, _dR2sldB1dTRF)
end