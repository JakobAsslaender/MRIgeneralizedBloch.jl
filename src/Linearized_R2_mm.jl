"""
    precompute_R2_mm_l([;TRF_min=100e-6, TRF_max=500e-6, T2_mm_min=5e-6, T2_mm_max=21e-6, ω1_max=π/TRF_max, B1_max=1.4, greens=greens_superlorentzian])

Pre-compute and interpolate the linearized `R2sl(TRF, α, B1, T2_mm)` and its derivatives `dR2sldB1(TRF, α, B1, T2_mm)`, `R2sldT2_mm(TRF, α, B1, T2_mm)` etc. in the range specified by the arguments.

The function solves the generalized Bloch equations of an isolated semi-solid pool for values in the specified range, calculates the linearized R2sl that minimizes the error of `zs` at the end of the RF-pulse, and interpolates between the different samples.

# Optional Arguments:
- `TRF_min::Real`: lower bound of the RF-pulse duration range in seconds
- `TRF_max::Real`: upper bound of the RF-pulse duration range in seconds
- `T2_mm_min::Real`: lower bound of the `T2_mm` range in seconds
- `T2_mm_max::Real`: upper bound of the `T2_mm` range in seconds
- `ω1_max::Real`: upper bound of the Rabi frequency ω1, the default is the frequency of a 500μs long π-pulse
- `B1_max::Real`: upper bound of the B1 range, normalized so that `B1 = 1` corresponds to a perfectly calibrated RF field
- `greens=greens_superlorentzian`: Greens function in the form `G(κ) = G((t-τ)/T2_mm)`. This package supplies the three Greens functions `greens=greens_superlorentzian` (default), `greens=greens_lorentzian`, and `greens=greens_gaussian`

# Examples
```jldoctest
julia> R2_mm_lT = precompute_R2_mm_l();


julia> R2_mm_l, dR2_mm_ldB1, R2_mm_ldT2_mm, _ = precompute_R2_mm_l(TRF_min=100e-6, TRF_max=500e-6, T2_mm_min=5e-6, T2_mm_max=15e-6, ω1_max=π/500e-6, B1_max=1.4, greens=greens_gaussian);

```
"""
function precompute_R2_mm_l(;TRF_min=100e-6, TRF_max=500e-6, T2_mm_min=5e-6, T2_mm_max=21e-6, ω1_max=π/TRF_max, B1_max=1.4, greens=greens_superlorentzian)

    # interpolate super-Lorentzian Green's function for speed purposes
    if greens == greens_superlorentzian
        G = interpolate_greens_function(greens, 0, TRF_max / T2_mm_min)
    else
        G = greens
    end

    function calculate_z(τ, Ω, G)
        mfun(p, t) = [1.0]

        function hamiltonian_1D!(du, u, mfun, p::NTuple{3,Any}, t)
            ω1, T2_mm, g = p
            du[1] = -ω1^2 * quadgk(x -> g((t - x) / T2_mm) * mfun(p, x)[1], 0.0, t)[1]
        end

        z = solve(DDEProblem(hamiltonian_1D!, [1.0], mfun, (0, τ), (Ω, 1, G)), MethodOfSteps(DP8()))
        return z
    end

    function calculate_R2_mm_l(z_fun, τ, Ω)
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

        sol = nlsolve(f!, j!, [0.15])
        return (sol.f_converged && sol.zero[1] > 0) ? sol.zero[1] : NaN
    end

    τv = range(TRF_min / T2_mm_max, TRF_max / T2_mm_min; length=2^10)
    Ωv = range(0, B1_max * ω1_max * T2_mm_max; length=2^8)

    A = Matrix{Float64}(undef, length(τv), length(Ωv))
    Threads.@threads for iΩ ∈ 2:length(Ωv)
        z_fun = calculate_z(τv[end], Ωv[iΩ], G)
        for iτ in eachindex(τv)
            A[iτ,iΩ] = calculate_R2_mm_l(z_fun, τv[iτ], Ωv[iΩ])
        end
    end
    A[:,1] .= A[:,2] # extrapolation hack as the fit does not work with Ω = 0

    f = linear_interpolation((τv, Ωv), A)
    dfdτ(   τ, Ω) = Interpolations.gradient(f, τ, Ω)[1]
    dfdΩ(   τ, Ω) = Interpolations.gradient(f, τ, Ω)[2]
    d2fdτ2( τ, Ω) = Interpolations.hessian( f, τ, Ω)[1,1]
    d2fdΩ2( τ, Ω) = Interpolations.hessian( f, τ, Ω)[2,2]
    d2fdτdΩ(τ, Ω) = Interpolations.hessian( f, τ, Ω)[2,1]

    R2_mm_l(TRF, α, B1, T2_mm) = f(TRF / T2_mm, B1 * α * T2_mm / TRF) / T2_mm

    dR2_mm_ldT2_mm(TRF, α, B1, T2_mm) = -dfdτ(TRF / T2_mm, B1 * α * T2_mm / TRF) * TRF / T2_mm^3 - f(TRF / T2_mm, B1 * α * T2_mm / TRF) / T2_mm^2 + dfdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) * B1 * α / TRF / T2_mm
    dR2_mm_ldB1(TRF, α, B1, T2_mm)  = dfdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) * α / TRF

    dR2_mm_ldω1(TRF, α, B1, T2_mm)  = dfdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) * B1
    dR2_mm_ldTRF(TRF, α, B1, T2_mm) = dfdτ(TRF / T2_mm, B1 * α * T2_mm / TRF) / T2_mm^2


    dR2_mm_ldT2_mmdω1(TRF, α, B1, T2_mm) = -d2fdτdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) * B1 * TRF / T2_mm^2 + d2fdΩ2(TRF / T2_mm, B1 * α * T2_mm / TRF) * B1^2 * α / TRF


    dR2_mm_ldB1dω1(TRF, α, B1, T2_mm) = dfdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) + d2fdΩ2(TRF / T2_mm, B1 * α * T2_mm / TRF) * B1 * α * T2_mm  / TRF
    dR2_mm_ldT2_mmTRF(TRF, α, B1, T2_mm) = -dfdτ(TRF / T2_mm, B1 * α * T2_mm / TRF) / T2_mm^3 - d2fdτ2(TRF / T2_mm, B1 * α * T2_mm / TRF) * TRF / T2_mm^4 - dfdτ(TRF / T2_mm, B1 * α * T2_mm / TRF) / T2_mm^3 + d2fdτdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) * B1 * α / TRF / T2_mm^2
    dR2_mm_ldB1dTRF(TRF, α, B1, T2_mm) = d2fdτdΩ(TRF / T2_mm, B1 * α * T2_mm / TRF) * α / TRF / T2_mm

    return (R2_mm_l, dR2_mm_ldT2_mm, dR2_mm_ldB1, dR2_mm_ldω1, dR2_mm_ldTRF, dR2_mm_ldT2_mmdω1, dR2_mm_ldB1dω1, dR2_mm_ldT2_mmTRF, dR2_mm_ldB1dTRF)
end


function evaluate_R2_mm_l_vector(α, TRF, B1, T2_mm, R2_mm_lT, grad_list)
    _R2_mm_ = similar(α)
    _dR2_mm_dT2_mm = similar(α)
    _dR2_mm_dB1 = similar(α)

    for i = 1:length(α)
        _R2_mm_[i] = R2_mm_lT[1](TRF[i], α[i], B1, T2_mm)
    end
    if any(isa.(grad_list, grad_T2_mm))
        for i = 1:length(α)
            _dR2_mm_dT2_mm[i] = R2_mm_lT[2](TRF[i], α[i], B1, T2_mm)
        end
    end
    if any(isa.(grad_list, grad_B1))
        for i = 1:length(α)
            _dR2_mm_dB1[i] = R2_mm_lT[3](TRF[i], α[i], B1, T2_mm)
        end
    end
    return (_R2_mm_, _dR2_mm_dT2_mm, _dR2_mm_dB1)
end