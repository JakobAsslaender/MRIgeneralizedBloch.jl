function precompute_R2sl(TRF_min, TRF_max, T2s_min, T2s_max, α_min, α_max, B1_min, B1_max)

    # approximate saturation
    G = interpolate_greens_function(greens_superlorentzian, 0, TRF_max / T2s_min)
    h(p, t) = [1.0]

    function hamiltonian_1D!(du, u, h, p::NTuple{3,Any}, t)
        ω1, T2s, g = p
        du[1] = -ω1^2 * quadgk(x -> g((t - x) / T2s) * h(p, x)[1], 0.0, t)[1]
    end

    function calculate_R2sl(τ, α)
        z = solve(DDEProblem(hamiltonian_1D!, [1.0], h, (0, τ), (α/τ, 1, G)), MethodOfSteps(DP8()))[end][1]
    
        function f!(F, ρv)
            ρ = ρv[1]
            s = sqrt(Complex((ρ*τ)^2 - (2α)^2))
            x = exp(-ρ*τ/2) * (cosh(s/2) + ρ*τ/s * sinh(s/2))
            F[1] = real(x) - z
        end
        function j!(J, ρv)
            ρ = ρv[1]
            s = sqrt(Complex((ρ*τ)^2 - (2α)^2))
            J[1] = 2τ * α^2 * exp(-ρ*τ/2) * (s * cosh(s/2) - 2sinh(s/2)) / s^3
        end
    
        sol = nlsolve(f!, j!, [1.0])
        return sol.zero[1]
    end

    τv = range(TRF_min/T2s_max, TRF_max/T2s_min; length=2^6)
    αv = range(B1_min * α_min, B1_max * α_max; length=2^6)
    A = [calculate_R2sl(τ, α) for τ in τv, α in αv]
    fapprox = CubicSplineInterpolation((τv, αv), A)

    dfd1(τ, α) = Interpolations.gradient(fapprox, τ, α)[1]
    dfd2(τ, α) = Interpolations.gradient(fapprox, τ, α)[2]
                                            
    function R2sl(TRF, ω1, B1, T2s)
        return fapprox(TRF / T2s, B1 * ω1 * TRF) / T2s
    end

    function R2sl_dB1(TRF, ω1, B1, T2s)
        _R2sl = fapprox(TRF / T2s, B1 * ω1 * TRF) / T2s
        _dR2sldB1 = dfd2(TRF / T2s, B1 * ω1 * TRF) * ω1 * TRF / T2s
        return (_R2sl, _dR2sldB1)
    end

    function R2sl_dB1_dT2s(TRF, ω1, B1, T2s)
        _fapprox = fapprox(TRF / T2s, B1 * ω1 * TRF)
        _dfd1 = dfd1(TRF / T2s, B1 * ω1 * TRF)
        _dfd2 = dfd2(TRF / T2s, B1 * ω1 * TRF)

        _R2sl = _fapprox / T2s
        _dR2sldB1 = _dfd2 * ω1 * TRF / T2s
        _R2sldT2s = - _dfd1 * TRF / T2s^3 - _fapprox / T2s^2
        return (_R2sl, _R2sldT2s, _dR2sldB1)
    end

    return (R2sl, R2sl_dB1, R2sl_dB1_dT2s)
end


function evaluate_R2sl_vector(ω1, TRF, B1, T2s, R2s_T, grad_list)
    _R2s = similar(ω1)
    _dR2sdT2s = similar(ω1)
    _dR2sdB1 = similar(ω1)

    if any(isa.(grad_list, grad_T2s))
        for i = 1:length(ω1)
            _R2s[i], _dR2sdT2s[i], _dR2sdB1[i] = R2s_T[3](TRF[i], ω1[i], B1, T2s)
        end
    elseif any(isa.(grad_list, grad_B1))
        for i = 1:length(ω1)
            _R2s[i], _dR2sdB1[i] = R2s_T[2](TRF[i], ω1[i], B1, T2s)
        end
    else
        for i = 1:length(ω1)
            _R2s[i] = R2s_T[1](TRF[i], ω1[i], B1, T2s)
        end
    end
    return (_R2s, _dR2sdT2s, _dR2sdB1)
end
