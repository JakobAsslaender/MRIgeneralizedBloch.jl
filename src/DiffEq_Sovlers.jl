"""
    calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s[; grad_list, Ncyc=2, output=:complexsignal])

Calculate the signal or magnetixation evolution with the full generalized Bloch model assuming a super-Lorentzian lineshape (slow).

The simulation assumes a sequence of rectangluar RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments. 

# Arguments
- `α::Vector{<:Number}`: Array of flip angles in radians
- `TRF::Vector{<:Number}`: Array of the RF-pulse durations in seconds
- `TR::Number`: Repetition time in seconds
- `ω0::Number`: Off-resonance frequency in rad/s 
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Number`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Number`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Number`: Transversal relaxationt time of the semi-solid pool in seconds

Optional:
- `grad_list=[]`: Vector to indicate which gradients should be calculated; the vector can either be empty `[]` for no gradient, or contain any subset/order of `grad_list=[grad_m0s(), grad_R1s(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `Ncyc=2`: The magnetization is initialized with thermal equilibrium and then performed Ncyc times and only the last cycle is stored. The default value is usually a good approximation for antiperiodic boundary conditions. Increase the number for higher precision at the cost of computation time. 
- `output=:complexsignal`: The defaul keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`
- `greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian)`: Tuple of a Greens function `G(κ) = G((t-τ)/T2s)` and its partial derivative wrt. T2s, multiplied by T2s `∂G((t-τ)/T2s)/∂T2s * T2s`. This package supplies the three Greens functions `greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian)` (default), `greens=(greens_lorentzian, dG_o_dT2s_x_T2s_lorentzian)`, and `greens=(greens_gaussian, dG_o_dT2s_x_T2s_gaussian)`

# Examples
```jldoctest
julia> calculatesignal_gbloch_ide(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.2, 0.3, 15, 20, 2, 10e-6)
100×1 Matrix{ComplexF64}:
   -0.00796631644501679 + 0.0im
  0.0012590590419314775 - 0.0im
  -0.006088855588122639 + 0.0im
  0.0024187389404174958 - 0.0im
  -0.004361339395232399 + 0.0im
   0.003489135821004988 - 0.0im
 -0.0027633710614651317 + 0.0im
   0.004483217941394422 - 0.0im
 -0.0012812573517352534 + 0.0im
    0.00540885403467719 - 0.0im
                        ⋮
   0.017760808273049045 - 0.0im
   0.017576118974646872 + 0.0im
   0.017813950945910605 - 0.0im
    0.01764385633550697 + 0.0im
   0.017863575855931853 - 0.0im
    0.01770692603385291 + 0.0im
   0.017909914934082043 - 0.0im
    0.01776565037329648 + 0.0im
   0.017953184893717555 - 0.0im

julia> calculatesignal_gbloch_ide(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.2, 0.3, 15, 20, 2, 10e-6; grad_list=[grad_R1f(), grad_T2s()], output=:realmagnetization)
100×15 transpose(::Matrix{Float64}) with eltype Float64:
 -0.00796627   0.0   0.000637773  …   0.0   -10.8757  -335.26   0.0
  0.00125903  -0.0  -0.00700671      -0.0   125.882   -326.977  0.0
 -0.00608882   0.0   0.00185086       0.0   -30.4187  -317.56   0.0
  0.00241873  -0.0  -0.00520622      -0.0    96.1776  -309.906  0.0
 -0.00436133   0.0   0.00296471       0.0   -47.5803  -302.948  0.0
  0.003489    -0.0  -0.00354518   …  -0.0    69.5148  -298.697  0.0
 -0.00276366   0.0   0.00399588       0.0   -62.8453  -294.886  0.0
  0.00448273  -0.0  -0.00200673      -0.0    45.3179  -292.783  0.0
 -0.00128187   0.0   0.00495478       0.0   -76.6573  -290.321  0.0
  0.00540814  -0.0  -0.000578836     -0.0    23.1756  -289.245  0.0
  ⋮                               ⋱
  0.0177563   -0.0   0.0175372       -0.0  -290.779   -349.855  0.0
  0.0175716    0.0   0.0177845        0.0  -295.347   -350.002  0.0
  0.0178094   -0.0   0.0176073       -0.0  -292.44    -350.163  0.0
  0.0176393    0.0   0.0178359        0.0  -296.668   -350.3    0.0
  0.017859    -0.0   0.0176727    …  -0.0  -294.001   -350.451  0.0
  0.0177024    0.0   0.0178838        0.0  -297.914   -350.579  0.0
  0.0179053   -0.0   0.0177335       -0.0  -295.467   -350.72   0.0
  0.0177611    0.0   0.0179286        0.0  -299.09    -350.84   0.0
  0.0179486   -0.0   0.0177902       -0.0  -296.845   -350.972  0.0
```
"""
function calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s; grad_list=[], Ncyc=2, output=:complexsignal, greens=(greens_superlorentzian, dG_o_dT2s_x_T2s_superlorentzian))
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
    alg = MethodOfSteps(DP8())
    s = zeros(N_s, length(TRF))
    m0 = zeros(N_s)
    m0[3] = (1 - m0s)
    m0[4] = m0s
    m0[5] = 1
    mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

    # prep pulse 
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G, dG_o_dT2s_x_T2s, grad_list)), alg)  
    m0 = sol[end]
    
    T_FP = (TR - TRF[2]) / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0, T_FP), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5())
    m0 = sol[end]
    
    for ic = 0:(Ncyc - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0, TRF[1] / 2), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5())
        m0 = sol[end]
        
        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = m0[1:3]
        m0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(DDEProblem(apply_hamiltonian_gbloch_inversion!, m0, mfun, (0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, m0s, 0, 0, 0, 0, T2s, G, dG_o_dT2s_x_T2s, grad_list)), alg)
        m0[4:5:end] = sol[end][4:5:end]

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
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        m0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G, dG_o_dT2s_x_T2s, grad_list)), alg)
            m0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            m0 = sol[end]
        end
    end
    s = transpose(s)
    if output == :complexsignal
        s = s[:, 1:5:end] + 1im * s[:, 2:5:end]
    end
    return s
end

"""
    calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s[; grad_list, Ncyc=2, output=:complexsignal])

Calculate the signal or magnetixation evolution with Graham's spectral model assuming a super-Lorentzian lineshape.

The simulation assumes a sequence of rectangluar RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments. 

# Arguments
- `α::Vector{<:Number}`: Array of flip angles in radians
- `TRF::Vector{<:Number}`: Array of the RF-pulse durations in seconds
- `TR::Number`: Repetition time in seconds
- `ω0::Number`: Off-resonance frequency in rad/s 
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Number`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Number`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Number`: Transversal relaxationt time of the semi-solid pool in seconds

Optional:
- `grad_list=[]`: Vector to indicate which gradients should be calculated; the vector can either be empty `[]` for no gradient, or contain any subset/order of `grad_list=[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1(), grad_R1a()]`
- `Ncyc=2`: The magnetization is initialized with thermal equilibrium and then performed Ncyc times and only the last cycle is stored. The default value is usually a good approximation for antiperiodic boundary conditions. Increase the number for higher precision at the cost of computation time. 
- `output=:complexsignal`: The defaul keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`

# Examples
```jldoctest
julia> calculatesignal_graham_ode(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.2, 0.3, 15, 20, 2, 10e-6)
100×1 Matrix{ComplexF64}:
  -0.00807345119202598 + 0.0im
 0.0012686432904785246 - 0.0im
 -0.006178694437440372 + 0.0im
  0.002435865817868477 - 0.0im
 -0.004437476277388273 + 0.0im
  0.003516464650880384 - 0.0im
 -0.002831554256573095 + 0.0im
  0.004523902251349811 - 0.0im
 -0.001342229952563084 + 0.0im
  0.005454562035727423 - 0.0im
                       ⋮
  0.018148222055411015 - 0.0im
  0.017957696642062964 + 0.0im
  0.018204860894110032 - 0.0im
  0.018029363128597056 + 0.0im
  0.018257820158882408 - 0.0im
   0.01809616895266688 + 0.0im
  0.018307337749218777 - 0.0im
   0.01815844455461564 + 0.0im
  0.018353636251669653 - 0.0im

julia> calculatesignal_graham_ode(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.2, 0.3, 15, 20, 2, 10e-6; grad_list=[grad_R1f(), grad_T2s()], output=:realmagnetization)
100×15 transpose(::Matrix{Float64}) with eltype Float64:
 -0.0080756    0.0   0.000643162  …   0.0   -10.4986  -323.634  0.0
  0.00126867  -0.0  -0.00710692      -0.0   123.078   -316.358  0.0
 -0.00618067   0.0   0.00186482       0.0   -29.4458  -307.862  0.0
  0.00243634  -0.0  -0.0052899       -0.0    94.1692  -300.821  0.0
 -0.00443746   0.0   0.00298646       0.0   -46.1422  -294.116  0.0
  0.00351386  -0.0  -0.00361718   …  -0.0    68.262   -289.421  0.0
 -0.00282882   0.0   0.00402793       0.0   -61.0236  -285.242  0.0
  0.00451808  -0.0  -0.00206741      -0.0    44.7888  -282.867  0.0
 -0.00133608   0.0   0.00499316       0.0   -74.4075  -280.79   0.0
  0.00544896  -0.0  -0.00062663      -0.0    23.2536  -280.148  0.0
  ⋮                               ⋱
  0.0181755   -0.0   0.0179431       -0.0  -285.071   -337.48   0.0
  0.0179911    0.0   0.0181963        0.0  -289.591   -337.634  0.0
  0.0182343   -0.0   0.018019        -0.0  -286.777   -337.804  0.0
  0.0180645    0.0   0.0182531        0.0  -290.964   -337.948  0.0
  0.0182893   -0.0   0.0180898    …  -0.0  -288.383   -338.106  0.0
  0.0181329    0.0   0.0183062        0.0  -292.261   -338.241  0.0
  0.0183407   -0.0   0.0181559       -0.0  -289.895   -338.39   0.0
  0.0181968    0.0   0.0183559        0.0  -293.488   -338.516  0.0
  0.0183888   -0.0   0.0182175       -0.0  -291.318   -338.656  0.0
```
"""
function calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s; grad_list=[], Ncyc=2, output=:complexsignal)
    ω1 = α ./ TRF

    # initialization and memory allocation 
    N_s = 5 * (1 + length(grad_list))
    s = zeros(N_s, length(TRF))
    m0 = zeros(N_s)
    m0[3] = (1 - m0s)
    m0[4] = m0s
    m0[5] = 1.0

    # prep pulse 
    sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, TRF[2], m0s, R1f, R2f, Rx, R1s, T2s, grad_list)), Tsit5(), save_everystep=false)    
    m0 = sol[end]
    
    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5(), save_everystep=false)
    m0 = sol[end]
    
    for ic = 0:(Ncyc - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, TRF[1] / 2), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5(), save_everystep=false)
        m0 = sol[end]

        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = m0[1:3]
        m0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        m0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian_inversionpulse!, m0, (0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, TRF[1], m0s, 0, 0, 0, 0, T2s, grad_list)), Tsit5(), save_everystep=false)
        m0[4:5:end] = sol[end][4:5:end]

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
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        m0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, TRF[ip], m0s, R1f, R2f, Rx, R1s, T2s, grad_list)), Tsit5(), save_everystep=false)
            m0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0, (0.0, T_FP), (ω0, m0s, R1f, R2f, Rx, R1s, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            m0 = sol[end]
        end
    end
    s = transpose(s)
    if output == :complexsignal
        s = s[:, 1:5:end] + 1im * s[:, 2:5:end]
    end
    return s
end