"""
    calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s[; grad_list, Ncyc=2, output=:complexsignal])

Calculate the signal or magnetixation evolution with the full generalized Bloch model assuming a super-Lorentzian lineshape (slow).

The simulation assumes a sequence of rectangluar RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments. 

# Arguemnts
- `α::Vector{<:Number}`: Array of flip angles in radians
- `TRF::Vector{<:Number}`: Array of the RF-pulse durations in seconds
- `TR::Number`: Repetition time in seconds
- `ω0::Number`: Off-resonance frequency in rad/s 
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1::Number`: Apparent longitudinal relaxation rate of the free and semi-solid pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `T2s::Number`: Transversal relaxationt time of the semi-solid pool in seconds

Optional:
- `grad_list=[]`: Vector to indicate which gradients should be calculated; the vector can either be empty `[]` for no gradient, or contain any subset/order of `grad_list=[grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]`
- `Ncyc=2`: The magnetization is initialized with thermal equilibrium and then performed Ncyc times and only the last cycle is stored. The default value is usually a good approximation for antiperiodic boundary conditions. Increase the number for higher precision at the cost of computation time. 
- `output=:complexsignal`: The defaul keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`
- `greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian)`: Tuple of a Greens function `G(κ) = G((t-τ)/T2s)` and its partial derivative wrt. T2s, multiplied by T2s `∂G((t-τ)/T2s)/∂T2s * T2s`. This package supplies the three Greens functions `greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian)` (default), `greens=(greens_lorentzian, dG_o_dT2s_x_T2s_lorentzian)`, and `greens=(greens_gaussian, dG_o_dT2s_x_T2s_gaussian)`

# Examples
```jldoctest
julia> calculatesignal_gbloch_ide(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6)
100×1 Matrix{ComplexF64}:
   -0.0246577624432389 + 0.0im
 0.0037348678313156953 - 0.0im
 -0.019057736704290798 + 0.0im
  0.007146413346945974 - 0.0im
 -0.013913423957603942 + 0.0im
  0.010291046550055142 - 0.0im
 -0.009153866379060188 + 0.0im
                       ⋮
   0.05283295983707664 + 0.0im
  0.053546314400920704 - 0.0im
   0.05303472239104932 + 0.0im
    0.0536945326257172 - 0.0im
   0.05322289237479402 + 0.0im
   0.05383318552722351 - 0.0im

julia> calculatesignal_gbloch_ide(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6; grad_list=[grad_R1(), grad_T2s()], output=:realmagnetization)
100×15 transpose(::Matrix{Float64}) with eltype Float64: [...]
```
"""
function calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s; grad_list=[], Ncyc=2, output=:complexsignal, greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian))
    ω1 = α ./ TRF

    # Define G(τ) and its derivative as ApproxFuns
    G = interpolate_greens_function(greens[1], 0, maximum(TRF) / T2s)
    if any(isa.(grad_list, grad_T2s))
        dG_o_dT2s_x_T2s = interpolate_greens_function(greens[2], 0, maximum(TRF) / T2s)
    else
        dG_o_dT2s_x_T2s = []
    end

    # initialization and memory allocation
    N_s = 5 * (1 + length(grad_list))
    h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(N_s)
    alg = MethodOfSteps(DP8())
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1

    # prep pulse 
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1, R2f, T2s, Rx, G, dG_o_dT2s_x_T2s, grad_list)), alg)  
    u0 = sol[end]
    
    T_FP = (TR - TRF[2]) / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5())
    u0 = sol[end]
    
    for ic = 0:(Ncyc - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0, TRF[1] / 2), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5())
        u0 = sol[end]
        
        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = u0[1:3]
        u0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(DDEProblem(apply_hamiltonian_gbloch_inversion!, u0, h, (0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, m0s, 0, 0, T2s, 0, G, dG_o_dT2s_x_T2s, grad_list)), alg)
        u0[4:5:end] = sol[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                u0[5i + 1] -= u00[1] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 2] += u00[2] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 3] -= u00[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        u0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1, R2f, T2s, Rx, G, dG_o_dT2s_x_T2s, grad_list)), alg)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    s = transpose(s)
    if output == :complexsignal
        s = s[:, 1:5:end] + 1im * s[:, 2:5:end]
    end
    return s
end

"""
    calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s[; grad_list, Ncyc=2, output=:complexsignal])

Calculate the signal or magnetixation evolution with Graham's spectral model assuming a super-Lorentzian lineshape.

The simulation assumes a sequence of rectangluar RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments. 

# Arguemnts
- `α::Vector{<:Number}`: Array of flip angles in radians
- `TRF::Vector{<:Number}`: Array of the RF-pulse durations in seconds
- `TR::Number`: Repetition time in seconds
- `ω0::Number`: Off-resonance frequency in rad/s 
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1::Number`: Apparent longitudinal relaxation rate of the free and semi-solid pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `T2s::Number`: Transversal relaxationt time of the semi-solid pool in seconds

Optional:
- `grad_list=[]`: Vector to indicate which gradients should be calculated; the vector can either be empty `[]` for no gradient, or contain any subset/order of `grad_list=[grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]`
- `Ncyc=2`: The magnetization is initialized with thermal equilibrium and then performed Ncyc times and only the last cycle is stored. The default value is usually a good approximation for antiperiodic boundary conditions. Increase the number for higher precision at the cost of computation time. 
- `output=:complexsignal`: The defaul keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`

# Examples
```jldoctest
julia> calculatesignal_graham_ode(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6)
100×1 Matrix{ComplexF64}:
 -0.025070162833645118 + 0.0im
  0.003743068709932292 - 0.0im
 -0.019432211972761165 + 0.0im
 0.0071589222453838685 - 0.0im
 -0.014255325151364606 + 0.0im
  0.010307593338620523 - 0.0im
 -0.009486903618759682 + 0.0im
                       ⋮
  0.053991689550114345 + 0.0im
   0.05480524072601308 - 0.0im
  0.054181571009529604 + 0.0im
   0.05493412868432605 - 0.0im
   0.05435879806126339 + 0.0im
   0.05505494420579709 - 0.0im

julia> calculatesignal_graham_ode(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6; grad_list=[grad_R1(), grad_T2s()], output=:realmagnetization)
100×15 transpose(::Matrix{Float64}) with eltype Float64: [...]
```
"""
function calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s; grad_list=[], Ncyc=2, output=:complexsignal)
    ω1 = α ./ TRF

    # initialization and memory allocation 
    N_s = 5 * (1 + length(grad_list))
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1.0

    # prep pulse 
    sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, u0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, TRF[2], m0s, R1, R2f, T2s, Rx, grad_list)), Tsit5(), save_everystep=false)    
    u0 = sol[end]
    
    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Ncyc - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, TRF[1] / 2), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false)
        u0 = sol[end]

        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = u0[1:3]
        u0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(ODEProblem(Graham_Hamiltonian_superLorentzian_InversionPulse!, u0, (0.0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, TRF[1], m0s, 0.0, 0.0, T2s, 0.0, grad_list)), Tsit5(), save_everystep=false)
        u0[4:5:end] = sol[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                u0[5i + 1] -= u00[1] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 2] += u00[2] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 3] -= u00[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        u0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, TRF[ip], m0s, R1, R2f, T2s, Rx, grad_list)), Tsit5(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    s = transpose(s)
    if output == :complexsignal
        s = s[:, 1:5:end] + 1im * s[:, 2:5:end]
    end
    return s
end