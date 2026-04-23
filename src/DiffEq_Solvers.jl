"""
    calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, M0, m0s, R1f, R2f, Rex, R1s, T2s[; grad_list=nothing, Ncyc=2, output=:complexsignal])

Calculate the signal or magnetization evolution with the full generalized Bloch model assuming a super-Lorentzian lineshape (slow).

The simulation assumes a sequence of rectangular RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments.

Always returns a tuple `(signal, gradients)` where `signal` is a vector (for `output=:complexsignal`) or matrix (for `output=:realmagnetization`) scaled by `M0`, and `gradients` is a matrix with one column per entry in `grad_list`, or `nothing` if no gradients are requested. `M0` must be real-valued; for complex-valued M0 (e.g. to account for receive coil phase), simulate with `M0=1` and multiply the complex M0 afterward.

# Arguments
- `α::Vector{Real}`: Array of flip angles in radians
- `TRF::Vector{Real}`: Array of the RF-pulse durations in seconds
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `M0::Real`: Equilibrium magnetization (proton density) scaling factor
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `Rex::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds

# Optional Arguments:
- `grad_list=nothing`: Tuple that specifies the gradients that are calculated; the default `nothing` means no gradient, or pass any subset/order of `grad_list=(grad_M0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`.
- `Ncyc=2`: The magnetization is initialized with thermal equilibrium and then performed Ncyc times and only the last cycle is stored. The default value is usually a good approximation for antiperiodic boundary conditions. Increase the number for higher precision at the cost of computation time.
- `output=:complexsignal`: The default keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`
- `greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian)`: Tuple of a Greens function `G(κ) = G((t-τ)/T2s)` and its partial derivative wrt. T2s, multiplied by T2s `∂G((t-τ)/T2s)/∂T2s * T2s`. This package supplies the three Greens functions `greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian)` (default), `greens=(greens_lorentzian, dG_o_dT2s_x_T2s_lorentzian)`, and `greens=(greens_gaussian, dG_o_dT2s_x_T2s_gaussian)`

# Examples
```jldoctest
julia> s, g = calculatesignal_gbloch_ide(fill(π/2, 100), fill(5e-4, 100), 4e-3, 0, 1, 1, 0.2, 0.3, 15, 20, 2, 10e-6);

julia> typeof(s)
Vector{ComplexF64} (alias for Array{Complex{Float64}, 1})

julia> size(s)
(100,)

julia> typeof(g)
Nothing
```
"""
function calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, M0::Real, m0s, R1f, R2f, Rex, R1s, T2s; grad_list=nothing, Ncyc=2, output=:complexsignal, greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian))
    ω1 = α ./ TRF

    isnothing(grad_list) && (grad_list = ())

    # Define G(τ) and its derivative as ApproxFuns
    G = interpolate_greens_function(greens[1], 0, maximum(TRF) / T2s)
    if any(isa.(grad_list, grad_T2s))
        dG_o_dT2s_x_T2s = interpolate_greens_function(greens[2], 0, maximum(TRF) / T2s)
    else
        dG_o_dT2s_x_T2s = ()
    end

    # initialization and memory allocation
    N_s = 5 * (1 + length(grad_list))
    alg = MethodOfSteps(DP8())
    s = zeros(N_s, length(TRF))
    m0 = zeros(N_s)
    m0[3] = M0 * (1 - m0s)
    m0[4] = M0 * m0s
    m0[5] = M0
    # initialize grad_M0 partial derivatives of initial conditions
    for i in eachindex(grad_list)
        if isa(grad_list[i], grad_M0)
            m0[5i + 3] = (1 - m0s)
            m0[5i + 4] = m0s
            m0[5i + 5] = 1
        end
    end
    mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

    # prep pulse
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s, G, dG_o_dT2s_x_T2s, grad_list)), alg)
    m0 = sol.u[end]

    T_FP = (TR - TRF[2]) / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0, T_FP), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5())
    m0 = sol.u[end]

    for ic = 0:(Ncyc - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0, TRF[1] / 2), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5())
        m0 = sol.u[end]

        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = m0[1:3]
        m0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(DDEProblem(apply_hamiltonian_gbloch_inversion!, m0, mfun, (0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, m0s, 0, 0, 0, 0, T2s, G, dG_o_dT2s_x_T2s, grad_list)), alg)
        m0[4:5:end] = sol.u[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                m0[5i + 1] -= u00[1] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                m0[5i + 2] += u00[2] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                m0[5i + 3] -= u00[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol.u[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        m0 = sol.u[end]

        for ip = 2:length(TRF)
            sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s, G, dG_o_dT2s_x_T2s, grad_list)), alg)
            m0 = sol.u[end]

            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(ErrorException("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol.u[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            m0 = sol.u[end]
        end
    end

    s = transpose(s)
    if output == :complexsignal
        signal = s[:, 1] + 1im * s[:, 2]
        if isempty(grad_list)
            gradients = nothing
        else
            gradients = Array{ComplexF64}(undef, length(TRF), length(grad_list))
            for j in eachindex(grad_list)
                gradients[:, j] = s[:, 5j+1] + 1im * s[:, 5j+2]
            end
        end
    else
        signal = s[:, 1:5]
        if isempty(grad_list)
            gradients = nothing
        else
            gradients = Array{Float64}(undef, length(TRF), 5 * length(grad_list))
            for j in eachindex(grad_list)
                gradients[:, (5(j-1)+1):(5j)] = s[:, (5j+1):(5j+5)]
            end
        end
    end
    return signal, gradients
end

"""
    calculatesignal_graham_ode(α, TRF, TR, ω0, B1, M0, m0s, R1f, R2f, Rex, R1s, T2s[; grad_list=nothing, Ncyc=2, output=:complexsignal])

Calculate the signal or magnetization evolution with Graham's spectral model assuming a super-Lorentzian lineshape.

The simulation assumes a sequence of rectangular RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments.

Always returns a tuple `(signal, gradients)` where `signal` is a vector (for `output=:complexsignal`) or matrix (for `output=:realmagnetization`) scaled by `M0`, and `gradients` is a matrix with one column per entry in `grad_list`, or `nothing` if no gradients are requested. `M0` must be real-valued; for complex-valued M0 (e.g. to account for receive coil phase), simulate with `M0=1` and multiply the complex M0 afterward.

# Arguments
- `α::Vector{Real}`: Array of flip angles in radians
- `TRF::Vector{Real}`: Array of the RF-pulse durations in seconds
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `M0::Real`: Equilibrium magnetization (proton density) scaling factor
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `Rex::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds

Optional:
- `grad_list=nothing`: Tuple that specifies the gradients that are calculated; the default `nothing` means no gradient, or pass any subset/order of `grad_list=(grad_M0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`.
- `Ncyc=2`: The magnetization is initialized with thermal equilibrium and then performed Ncyc times and only the last cycle is stored. The default value is usually a good approximation for antiperiodic boundary conditions. Increase the number for higher precision at the cost of computation time.
- `output=:complexsignal`: The default keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`

# Examples
```jldoctest
julia> s, g = calculatesignal_graham_ode(fill(π/2, 100), fill(5e-4, 100), 4e-3, 0, 1, 1, 0.2, 0.3, 15, 20, 2, 10e-6);

julia> typeof(s)
Vector{ComplexF64} (alias for Array{Complex{Float64}, 1})

julia> size(s)
(100,)

julia> typeof(g)
Nothing
```
"""
function calculatesignal_graham_ode(α, TRF, TR, ω0, B1, M0::Real, m0s, R1f, R2f, Rex, R1s, T2s; grad_list=nothing, Ncyc=2, output=:complexsignal)
    ω1 = α ./ TRF

    isnothing(grad_list) && (grad_list = ())

    # initialization and memory allocation
    N_s = 5 * (1 + length(grad_list))
    s = zeros(N_s, length(TRF))
    m0 = zeros(N_s)
    m0[3] = M0 * (1 - m0s)
    m0[4] = M0 * m0s
    m0[5] = M0
    # initialize grad_M0 partial derivatives of initial conditions
    for i in eachindex(grad_list)
        if isa(grad_list[i], grad_M0)
            m0[5i + 3] = (1 - m0s)
            m0[5i + 4] = m0s
            m0[5i + 5] = 1
        end
    end

    # prep pulse
    sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, TRF[2], m0s, R1f, R2f, Rex, R1s, T2s, grad_list)), Tsit5(), save_everystep=false)
    m0 = sol.u[end]

    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5(), save_everystep=false)
    m0 = sol.u[end]

    for ic = 0:(Ncyc - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, TRF[1] / 2), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5(), save_everystep=false)
        m0 = sol.u[end]

        # inversion pulse with crusher gradients (assumed to be instantaneous)
        u00 = m0[1:3]
        m0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian_inversionpulse!, m0, (0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, TRF[1], m0s, 0, 0, 0, 0, T2s, grad_list)), Tsit5(), save_everystep=false)
        m0[4:5:end] = sol.u[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                m0[5i + 1] -= u00[1] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                m0[5i + 2] += u00[2] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                m0[5i + 3] -= u00[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol.u[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        m0 = sol.u[end]

        for ip = 2:length(TRF)
            sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, TRF[ip], m0s, R1f, R2f, Rex, R1s, T2s, grad_list)), Tsit5(), save_everystep=false)
            m0 = sol.u[end]

            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(ErrorException("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol.u[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            m0 = sol.u[end]
        end
    end

    s = transpose(s)
    if output == :complexsignal
        signal = s[:, 1] + 1im * s[:, 2]
        if isempty(grad_list)
            gradients = nothing
        else
            gradients = Array{ComplexF64}(undef, length(TRF), length(grad_list))
            for j in eachindex(grad_list)
                gradients[:, j] = s[:, 5j+1] + 1im * s[:, 5j+2]
            end
        end
    else
        signal = s[:, 1:5]
        if isempty(grad_list)
            gradients = nothing
        else
            gradients = Array{Float64}(undef, length(TRF), 5 * length(grad_list))
            for j in eachindex(grad_list)
                gradients[:, (5(j-1)+1):(5j)] = s[:, (5j+1):(5j+5)]
            end
        end
    end
    return signal, gradients
end