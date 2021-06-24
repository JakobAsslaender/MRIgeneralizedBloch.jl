###################################################
# generalized Bloch Hamiltonians that can take any 
# Green's function as an argument. 
###################################################
"""
    apply_hamiltonian_gbloch!(du, u, h, p, t)

Apply the generalized Bloch Hamiltonian to `u` and write the resulting derivative wrt. time into `du`.

# Arguemnts
- `du::Vector{<:Number}`: Array describing to derivative of u wrt. time; this vector has to be of the same size as `u`, but can contain any value, which is replaced by `H * u`
- `u::Vector{<:Number}`: Array the spin ensemble state of the form `[xf, yf, zf, zs, 1]` if now gradient is calculated or of the form `[xf, yf, zf, zs, 1, ∂xf/∂θ1, ∂yf/∂θ1, ∂zf/∂θ1, ∂zs/∂θ1, 0, ..., ∂xf/∂θn, ∂yf/∂θn, ∂zf/∂θn, ∂zs/∂θn, 0]` if n derivatives wrt. `θn` are calculated
- `h`: History fuction; can be initialized with `h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5n + 5)` for n gradients, and is then updated by the delay differential equation solvers
- `p::NTuple{9,10, or 11, Any}`: `(ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g)`, whith 
    -`ω1::Number`: Rabi frequency in rad/s (rotation about the y-axis)
    -`B1::Number`: B1 scaling normalized so that `B1=1` corresponds to a perfectly calibrated RF field
    -`ω0::Number`: Larmor or off-resonance frequency in rad/s
    -`m0s::Number`: Fractional semi-solid spin pool size in the range of 0 to 1
    -`R1::Number`: Apparent longitudinal spin relaxation rate of both pools in 1/seconds
    -`R2f::Number`: Trasversal spin relaxation rate of the free pool in 1/seconds
    -`T2s::Number`: Trasversal spin relaxation time of the semi-solid pool in seconds
    -`Rx::Number`: Exchange rate between the two pools in 1/seconds
    -`g::Function`: Green's function of the form `G(κ) = G((t-τ)/T2s)`
    or `(ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, zs_idx, g)` with
    - `zs_idx::Integer`: Index to be used history function to be used in the Green's function; Default is 4 (zs), and for derivatives 9, 14, ... are used
    or `(ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g, dG_o_dT2s_x_T2s, grad_list)` with
    - `dG_o_dT2s_x_T2s::Function`: Derivative of the Green's function wrt. T2s, multiplied by T2s; of the form `dG_o_dT2s_x_T2s(κ) = dG_o_dT2s_x_T2s((t-τ)/T2s)`
    - `grad_list::Vector{<:grad_param}`: List of gradients to be calucualted; any subset of `[grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]`; length of the vector must be n (cf. arguments `u` and `du`)
- `t::Number`: Time in seconds

Optional:
- `pulsetype=:normal`: Use default for a regular RF-pulse; the option `pulsetype=:inversion` should be handled with care as it is only inteded to calculate the saturation of the semi-solid pool and its derivative. 

# Examples
```jldoctest
julia> using DifferentialEquations
julia> α = π/2
julia> TRF = 100e-6
julia> ω1 = α/TRF
julia> B1 = 1
julia> ω0 = 0
julia> m0s = 0.1
julia> R1 = 1
julia> R2f = 15
julia> T2s = 10-6
julia> Rx = 30
julia> G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s)
julia> u0 = [0; 0; 1-m0s; m0s; 1]
julia> h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)
julia> sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0, TRF), (-ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, G)), MethodOfSteps(DP8()))  
retcode: Success
Interpolation: specialized 7th order interpolation
t: 6-element Vector{Float64}:
 0.0
 1.220281289257312e-7
 8.541969024801185e-7
 5.247209543806443e-6
 3.160528539176439e-5
 0.0001
u: 6-element Vector{Vector{Float64}}:
 [0.0, 0.0, 0.9, 0.1, 1.0]
 [-0.0017251293948764102, 0.0, 0.8999983466235149, 0.0999998162913902, 1.0]
 [-0.012075484505676836, 0.0, 0.8999189860592292, 0.09999099841258494, 1.0]
 [-0.07409379835058187, 0.0, 0.8969447197649016, 0.0996605155131504, 1.0]
 [-0.4285786292638301, 0.0, 0.79136732759974, 0.0879278039163368, 1.0]
 [-0.8993375887462968, 0.0, 0.0004582695875068321, 3.215359474631474e-6, 1.0]

 julia> using Plots
 julia> plot(sol, labels=["xf" "yf" "zf" "zs" "1"], xlabel="t (s)", ylabel="m(t)")

julia> dG_o_dT2s_x_T2s = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, TRF / T2s)
julia> grad_list = [grad_R2f(), grad_m0s()]
julia> u0 = [0; 0; 1-m0s; m0s; 1; zeros(5*length(grad_list))]
julia> h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5 + 5*length(grad_list))
julia> sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0, TRF), (-ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, G, dG_o_dT2s_x_T2s, grad_list)), MethodOfSteps(DP8()))  
retcode: Success
Interpolation: specialized 7th order interpolation
t: 6-element Vector{Float64}:
 0.0
 1.2202754217257472e-7
 8.541927952080231e-7
 5.247184313420713e-6
 3.1605133422696854e-5
 0.0001
u: 6-element Vector{Vector{Float64}}:
 [0.0, 0.0, 0.9, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 [...]

 julia> plot(sol)
```
"""
function apply_hamiltonian_gbloch!(du, u, h, p::NTuple{10,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, zs_idx, g = p

    du[1] = - R2f * u[1] - ω0  * u[2] + B1 * ω1 * u[3]
    du[2] =   ω0  * u[1] - R2f * u[2]
    du[3] = - B1 * ω1  * u[1] - (R1 + Rx * m0s) * u[3] + Rx * (1 - m0s) * u[4] + (1 - m0s) * R1 * u[5]
    du[4] = -B1^2 * ω1^2 * quadgk(x -> g((t - x) / T2s) * h(p, x; idxs=zs_idx), eps(), t)[1] + Rx * m0s  * u[3] - (R1 + Rx * (1 - m0s)) * u[4] + m0s * R1 * u[5]
    return du
end

function apply_hamiltonian_gbloch!(du, u, h, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g = p
    return apply_hamiltonian_gbloch!(du, u, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, 4, g), t)
end

function apply_hamiltonian_gbloch!(du, u, h, p::NTuple{11,Any}, t; pulsetype=:normal)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g, dG_o_dT2s_x_T2s, grad_list = p
    
    # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    apply_hamiltonian_gbloch!(du_v1, u_v1, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, 4, g), t)

    # Apply Hamiltonian to all derivatives and add partial derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_gbloch!(du_v, u_v, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, (5i + 4), g), t)

        if pulsetype==:normal || isa(grad_list[i], grad_T2s) || isa(grad_list[i], grad_B1)
            add_partial_derivative!(du_v, u_v1, x -> h(p, x; idxs=4), (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g, dG_o_dT2s_x_T2s), t, grad_list[i])
        end
    end
    return du
end

function apply_hamiltonian_gbloch_inversion!(du, u, h, p::NTuple{11,Any}, t)
    apply_hamiltonian_gbloch!(du, u, h, p::NTuple{11,Any}, t; pulsetype=:inversion)
end

###################################################
# Bloch-McConnel model to simulate free precession
###################################################
function apply_hamiltonian_freeprecession!(du, u, p::NTuple{5,Any}, t)
    ω0, m0s, R1, R2f, Rx = p

    du[1] = - R2f * u[1] - ω0  * u[2]
    du[2] =   ω0  * u[1] - R2f * u[2]
    du[3] = - (R1 + Rx * m0s) * u[3] + Rx * (1 - m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] =   Rx * m0s  * u[3] - (R1 + Rx * (1 - m0s)) * u[4] + m0s  * R1 * u[5]
    return du
end

function apply_hamiltonian_freeprecession!(du, u, p::NTuple{6,Any}, t)
    ω0, m0s, R1, R2f, Rx, grad_list = p

    # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    apply_hamiltonian_freeprecession!(du_v1, u_v1, (ω0, m0s, R1, R2f, Rx), t)

    # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_freeprecession!(du_v, u_v, (ω0, m0s, R1, R2f, Rx), t)

        add_partial_derivative!(du_v, u_v1, [], (0.0, 1.0, ω0, m0s, R1, R2f, [], Rx, [], []), t, grad_list[i])
    end
    return du
end

###################################################
# implementatoin of the partial derivates for 
# calculationg th gradient
###################################################
function add_partial_derivative!(du, u, h, p, t, grad_type::grad_m0s)
    _, _, _, _, R1, _, _, Rx, _, _ = p

    du[3] -= Rx * u[3] + Rx * u[4] + R1
    du[4] += Rx * u[3] + Rx * u[4] + R1
    return du
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_R1)
    _, _, _, m0s, _, _, _, _, _, _ = p

    du[3] += - u[3] + (1 - m0s)
    du[4] += - u[4] + m0s
    return du
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_R2f)
    du[1] -= u[1]
    du[2] -= u[2]
    return du
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_Rx)
    _, _, _, m0s, _, _, _, _, _, _ = p

    du[3] += - m0s * u[3] + (1 - m0s) * u[4]
    du[4] +=   m0s * u[3] - (1 - m0s) * u[4]
    return du
end

# version for gBloch with using ApproxFun
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Fun,Fun}, t, grad_type::grad_T2s)
    ω1, B1, _, _, _, _, T2s, _, _, dG_o_dT2s_x_T2s = p
    
    du[4] -= B1^2 * ω1^2 / T2s * quadgk(x -> dG_o_dT2s_x_T2s((t - x) / T2s) * h(x), 0.0, t)[1]
    return du
end

# version for free precession (does nothing)
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Array{Any,1},Array{Any,1}}, t, grad_type::grad_T2s)
    return du
end

# version for Graham's model
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Number,Any}, t, grad_type::grad_T2s)
    ω1, B1, _, _, _, _, T2s, _, TRF, _ = p
    
    df_PSD = (τ) -> quadgk(ct -> 8 / τ * (exp(-τ^2 / 8 * (3 * ct^2 - 1)^2) - 1) / (3 * ct^2 - 1)^2 + sqrt(2π) * erf(τ / sqrt(8) * abs(3 * ct^2 - 1)) / abs(3 * ct^2 - 1), 0.0, 1.0)[1]
        
    du[4] -= df_PSD(TRF / T2s) * B1^2 * ω1^2 * u[4]
    return du
end

# version for linearized gBloch
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Tuple,Any}, t, grad_type::grad_T2s)
    _, _, _, _, _, _, _, _, Rrf_d, _ = p
    
    du[4] -= Rrf_d[3] * u[4]
    return du
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_ω0)
    du[1] -= u[2]
    du[2] += u[1]
    return du
end

# version for gBloch (using ApproxFun)
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Fun,Any}, t, grad_type::grad_B1)
    ω1, B1, _, _, _, _, T2s, _, g, _ = p
    
    du[1] += ω1 * u[3]
    du[3] -= ω1 * u[1]
    du[4] -= 2 * B1 * ω1^2 * quadgk(x -> g((t - x) / T2s) * h(x), eps(), t)[1]
    return du
end

# version for free precession (does nothing)
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Array{Any,1},Array{Any,1}}, t, grad_type::grad_B1)
    return du
end

# version for Graham
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Number,Any}, t, grad_type::grad_B1)
    ω1, B1, _, _, _, _, T2s, _, TRF, _ = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0)[1]

    du[1] += ω1 * u[3]
    du[3] -= ω1 * u[1]
    du[4] -= f_PSD(TRF / T2s) * 2 * B1 * ω1^2 * T2s * u[4]
    return du
end

# version for linearized gBloch
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Tuple,Any}, t, grad_type::grad_B1)
    ω1, _, _, _, _, _, _, _, Rrf_d, _ = p

    du[1] += ω1 * u[3]
    du[3] -= ω1 * u[1]
    du[4] -= Rrf_d[2] * u[4]
    return du
end


###################################################
# Implementation for comparison: the super-Lorentzian 
# Green's function is hard coded, which allows to 
# use special solvers for the double integral
###################################################
function apply_hamiltonian_gbloch_superlorentzian!(du, u, h, p::NTuple{10,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, zs_idx, N = p

    gt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3 * ct^2 - 1)^2 / 8)

    function fy!(x, y, gt, h, p, T2s, zs_idx, t)
        for i = 1:size(x, 2)
            y[i] = gt(t - x[2,i], T2s, x[1,i]) * h(p, x[2,i]; idxs=zs_idx)
        end
    end

    dy1 = Cubature.pcubature_v((x, y) -> fy!(x, y, gt, h, p, T2s, zs_idx, t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = T2s * sqrt(2π / 3) * Cubature.pcubature(x -> h(p, x[1]; idxs=zs_idx) / (t - x[1]), [0.0], [t - N * T2s])[1]
        
        du[4] = -B1^2 * ω1^2 * ((dy1) + (dy2))
    else
        du[4] = -B1^2 * ω1^2 * (dy1)
    end

    du[1] = - R2f * u[1] - ω0  * u[2] + B1 * ω1 * u[3]
    du[2] =   ω0  * u[1] - R2f * u[2]
    du[3] = - B1 * ω1  * u[1] - (R1 + Rx * m0s) * u[3] +       Rx * (1 - m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] +=             +       Rx * m0s  * u[3] - (R1 + Rx * (1 - m0s)) * u[4] +      m0s  * R1 * u[5]
    du[5] = 0.0
    return du
end

function apply_hamiltonian_gbloch_superlorentzian!(du, u, h, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, N = p
    return apply_hamiltonian_gbloch_superlorentzian!(du, u, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, 4, N), t)
end

###################################################
# Graham's spectral model
###################################################
function apply_hamiltonian_graham_superlorentzian!(du, u, p::NTuple{9,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0)[1]
    Rrf = f_PSD(TRF / T2s) * B1^2 * ω1^2 * T2s

    return apply_hamiltonian_linear!(du, u, (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf), t)
end

function apply_hamiltonian_graham_superlorentzian!(du, u, p::NTuple{10,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx, grad_list = p
    
     # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    apply_hamiltonian_graham_superlorentzian!(du_v1, u_v1, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
     # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_graham_superlorentzian!(du_v, u_v, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
        add_partial_derivative!(du_v, u_v1, [], (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, TRF, []), t, grad_list[i])
    end
    return du
end

function Graham_Hamiltonian_superLorentzian_InversionPulse!(du, u, p::NTuple{10,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx, grad_list = p
    
     # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    apply_hamiltonian_graham_superlorentzian!(du_v1, u_v1, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
     # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_graham_superlorentzian!(du_v, u_v, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
        if isa(grad_list[i], grad_B1) || isa(grad_list[i], grad_T2s)
            add_partial_derivative!(du_v, u_v1, [], (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, TRF, []), t, grad_list[i])
        end
    end
    return du
end


function apply_hamiltonian_linear!(du, u, p::NTuple{8,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf = p
    
    apply_hamiltonian_freeprecession!(du, u, (ω0, m0s, R1, R2f, Rx), t)

    du[1] += B1 * ω1 * u[3]
    du[3] -= B1 * ω1 * u[1]
    du[4] -= Rrf * u[4]
    return du
end

function apply_hamiltonian_linear!(du, u, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf_d, grad_list = p
    Rrf = Rrf_d[1]
    
     # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    apply_hamiltonian_linear!(du_v1, u_v1, (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf), t)
 
     # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_linear!(du_v, u_v, (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf), t)
 
        add_partial_derivative!(du_v, u_v1, [], (ω1, B1, ω0, m0s, R1, R2f, 0.0, Rx, Rrf_d, []), t, grad_list[i])
    end
    return du
end