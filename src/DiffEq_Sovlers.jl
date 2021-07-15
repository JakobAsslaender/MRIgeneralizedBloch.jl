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
 -0.024657762441422173 + 0.0im
 0.0037348678313655374 - 0.0im
 -0.019057736703007203 + 0.0im
  0.007146413346758786 - 0.0im
 -0.013913423956595965 + 0.0im
  0.010291046549792262 - 0.0im
 -0.009153866378612974 + 0.0im
  0.013213045210360628 - 0.0im
 -0.004734258510785933 + 0.0im
   0.01593906991792926 - 0.0im
                       ⋮
  0.053218511651564396 - 0.0im
   0.05261662009092078 + 0.0im
   0.05338787446252438 - 0.0im
     0.052832959843115 + 0.0im
  0.053546314408472656 - 0.0im
   0.05303472239762076 + 0.0im
  0.053694532633733943 - 0.0im
   0.05322289238188523 + 0.0im
  0.053833185535691296 - 0.0im

julia> calculatesignal_gbloch_ide(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6; grad_list=[grad_R1(), grad_T2s()], output=:realmagnetization)
100×15 transpose(::Matrix{Float64}) with eltype Float64:
 -0.0246575    0.0   0.00191834  …   0.0    -8.09518  -146.358   0.0
  0.00373479  -0.0  -0.0217727      -0.0    83.521    -135.807   0.0
 -0.0190576    0.0   0.00549726      0.0   -21.4616   -119.477   0.0
  0.00714631  -0.0  -0.0164074      -0.0    64.4139   -106.11    0.0
 -0.0139133    0.0   0.0087715       0.0   -30.9968    -92.3956  0.0
  0.0102909   -0.0  -0.0114604   …  -0.0    49.4596    -83.7202  0.0
 -0.00915379   0.0   0.0118022       0.0   -37.5344    -75.2901  0.0
  0.0132129   -0.0  -0.00687472     -0.0    37.423     -71.2706  0.0
 -0.00473424   0.0   0.0146242       0.0   -42.288     -66.9254  0.0
  0.0159389   -0.0  -0.00261256     -0.0    27.1804    -66.0774  0.0
  ⋮                              ⋱
  0.0532167   -0.0   0.0525262      -0.0  -206.85     -155.403   0.0
  0.0526148    0.0   0.0533279       0.0  -210.517    -155.62    0.0
  0.053386    -0.0   0.0527502      -0.0  -209.167    -155.899   0.0
  0.0528312    0.0   0.0534917       0.0  -212.584    -156.103   0.0
  0.0535445   -0.0   0.052959    …  -0.0  -211.372    -156.364   0.0
  0.0530329    0.0   0.0536449       0.0  -214.557    -156.555   0.0
  0.0536927   -0.0   0.0531539      -0.0  -213.471    -156.799   0.0
  0.053221     0.0   0.0537883       0.0  -216.439    -156.979   0.0
  0.0538313   -0.0   0.0533355      -0.0  -215.468    -157.207   0.0
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
    alg = MethodOfSteps(DP8())
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1
    h(p, t; idxs=nothing) = typeof(idxs) <: Number ? u0[idxs] : u0

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
 -0.025070162833645618 + 0.0im
  0.003743068709932302 - 0.0im
 -0.019432211972761584 + 0.0im
  0.007158922245383879 - 0.0im
    -0.014255325151365 + 0.0im
  0.010307593338620604 - 0.0im
 -0.009486903618759873 + 0.0im
  0.013252701136887497 - 0.0im
  -0.00505007378048608 + 0.0im
  0.015978096974037573 - 0.0im
                       ⋮
    0.0545210792217067 - 0.0im
   0.05378824747233755 + 0.0im
   0.05466774677353861 - 0.0im
   0.05399168955011439 + 0.0im
   0.05480524072601252 - 0.0im
   0.05418157100952948 + 0.0im
   0.05493412868432476 - 0.0im
   0.05435879806126212 + 0.0im
  0.055054944205796146 - 0.0im

julia> calculatesignal_graham_ode(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6; grad_list=[grad_R1(), grad_T2s()], output=:realmagnetization)
100×15 transpose(::Matrix{Float64}) with eltype Float64:
 -0.0249614    0.0   0.00192295  …   0.0    -7.79895  -141.002   0.0
  0.00374263  -0.0  -0.0220613      -0.0    81.4297   -131.309   0.0
 -0.0193318    0.0   0.00550778      0.0   -20.7551   -116.146   0.0
  0.00715882  -0.0  -0.0166685      -0.0    62.8368   -103.541   0.0
 -0.014162     0.0   0.00878636      0.0   -30.0836    -90.4788  0.0
  0.010306    -0.0  -0.0117047   …  -0.0    48.2494    -81.8859  0.0
 -0.00938721   0.0   0.0118269       0.0   -36.5357    -73.5855  0.0
  0.0132393   -0.0  -0.00710492     -0.0    36.5226    -69.4018  0.0
 -0.0049538    0.0   0.0146514       0.0   -41.1928    -65.1257  0.0
  0.0159661   -0.0  -0.00282248     -0.0    26.5349    -64.1602  0.0
  ⋮                              ⋱
  0.0541053   -0.0   0.053379       -0.0  -202.073    -149.963   0.0
  0.0535064    0.0   0.0541954       0.0  -205.711    -150.191   0.0
  0.054291    -0.0   0.0536194      -0.0  -204.431    -150.476   0.0
  0.0537389    0.0   0.054375        0.0  -207.825    -150.691   0.0
  0.054465    -0.0   0.0538441   …  -0.0  -206.681    -150.958   0.0
  0.0539561    0.0   0.0545433       0.0  -209.847    -151.16    0.0
  0.0546281   -0.0   0.0540539      -0.0  -208.826    -151.411   0.0
  0.054159     0.0   0.054701        0.0  -211.78     -151.601   0.0
  0.0547809   -0.0   0.05425        -0.0  -210.87     -151.835   0.0
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
        sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian_inversionpulse!, u0, (0.0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, TRF[1], m0s, 0.0, 0.0, T2s, 0.0, grad_list)), Tsit5(), save_everystep=false)
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