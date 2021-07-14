# todo: replace h with m_fun
###################################################
# generalized Bloch Hamiltonians that can take any 
# Green's function as an argument. 
###################################################
"""
    apply_hamiltonian_gbloch!(∂m∂t, m, h, p, t)

Apply the generalized Bloch Hamiltonian to `m` and write the resulting derivative wrt. time into `∂m∂t`.

# Arguemnts
- `∂m∂t::Vector{<:Number}`: Vector describing to derivative of `m` wrt. time; this vector has to be of the same size as `m`, but can contain any value, which is replaced by `H * m`
- `m::Vector{<:Number}`: Vector the spin ensemble state of the form `[xf, yf, zf, zs, 1]` if now gradient is calculated or of the form `[xf, yf, zf, zs, 1, ∂xf/∂θ1, ∂yf/∂θ1, ∂zf/∂θ1, ∂zs/∂θ1, 0, ..., ∂xf/∂θn, ∂yf/∂θn, ∂zf/∂θn, ∂zs/∂θn, 0]` if n derivatives wrt. `θn` are calculated
- `h`: History fuction; can be initialized with `h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5n + 5)` for n gradients, and is then updated by the delay differential equation solvers
- `p::NTuple{9,10, or 11, Any}`: `(ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g)`, with 
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
    - `grad_list::Vector{<:grad_param}`: List of gradients to be calucualted; any subset of `[grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]`; length of the vector must be n (cf. arguments `m` and `∂m∂t`)
- `t::Number`: Time in seconds

Optional:
- `pulsetype=:normal`: Use default for a regular RF-pulse; the option `pulsetype=:inversion` should be handled with care as it is only inteded to calculate the saturation of the semi-solid pool and its derivative. 

# Examples
```jldoctest
julia> using DifferentialEquations

julia> α = π/2;

julia> TRF = 100e-6;

julia> ω1 = α/TRF;

julia> B1 = 1;

julia> ω0 = 0;

julia> m0s = 0.1;

julia> R1 = 1;

julia> R2f = 15;

julia> T2s = 10e-6;

julia> Rx = 30;

julia> G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);


julia> u0 = [0; 0; 1-m0s; m0s; 1];

julia> h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5);

julia> sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0, TRF), (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, G)), MethodOfSteps(DP8()))
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
 [0.0017251293948764097, 0.0, 0.8999983466235149, 0.09999981629184612, 1.0]
 [0.012075484505676845, 0.0, 0.8999189860592341, 0.09999099950691269, 1.0]
 [0.07409379835118453, 0.0, 0.8969447198086778, 0.09966205538313169, 1.0]
 [0.42857865289497454, 0.0, 0.7913675966569867, 0.08937976862223226, 1.0]
 [0.8993472985166022, 0.0, 0.0004869430378807708, 0.04106841741188661, 1.0]

julia> using Plots

julia> plot(sol, labels=["xf" "yf" "zf" "zs" "1"], xlabel="t (s)", ylabel="m(t)");


julia> dG_o_dT2s_x_T2s = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, TRF / T2s);


julia> grad_list = [grad_R2f(), grad_m0s()];


julia> u0 = [0; 0; 1-m0s; m0s; 1; zeros(5*length(grad_list))];


julia> h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5 + 5*length(grad_list));

julia> sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0, TRF), (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, G, dG_o_dT2s_x_T2s, grad_list)), MethodOfSteps(DP8()));


julia> plot(sol);
```
"""
function apply_hamiltonian_gbloch!(∂m∂t, m, h, p::NTuple{10,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, zs_idx, g = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1 * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1  * m[1] - (R1 + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1 * m[5]

    if ω0 == 0
        xs = 0
        ys = quadgk(x -> g((t - x) / T2s) * h(p, x; idxs=zs_idx), eps(), t, order=100)[1]
    else
        xs = sin(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(p, x; idxs=zs_idx) * sin(ω0 * x), eps(), t, order=100)[1]
        ys = cos(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(p, x; idxs=zs_idx) * cos(ω0 * x), eps(), t, order=100)[1]
    end

    ∂m∂t[4] = -B1^2 * ω1^2 * (xs + ys) + Rx * m0s  * m[3] - (R1 + Rx * (1 - m0s)) * m[4] + m0s * R1 * m[5]
    return ∂m∂t
end

function apply_hamiltonian_gbloch!(∂m∂t, m, h, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g = p
    return apply_hamiltonian_gbloch!(∂m∂t, m, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, 4, g), t)
end

# Version for an isolated semi-solid pool 
function apply_hamiltonian_gbloch!(∂m∂t, m, h, p::NTuple{6,Any}, t)
    ω1, B1, ω0, R1, T2s, g = p
    
    yt = cos(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(p, x)[1] * cos(ω0 * x), 0, t, order=100)[1]
    xt = sin(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(p, x)[1] * sin(ω0 * x), 0, t, order=100)[1]
    
    ∂m∂t[1] = -B1^2 * ω1^2 * (xt + yt) + R1 * (1 - m[1])
end


function apply_hamiltonian_gbloch!(∂m∂t, m, h, p::NTuple{11,Any}, t; pulsetype=:normal)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g, dG_o_dT2s_x_T2s, grad_list = p
    
    # Apply Hamiltonian to M
    u_v1 = @view m[1:5]
    du_v1 = @view ∂m∂t[1:5]
    apply_hamiltonian_gbloch!(du_v1, u_v1, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, 4, g), t)

    # Apply Hamiltonian to all derivatives and add partial derivatives
    for i = 1:length(grad_list)
        du_v = @view ∂m∂t[5 * i + 1:5 * (i + 1)]
        u_v  = @view m[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_gbloch!(du_v, u_v, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, (5i + 4), g), t)

        if pulsetype==:normal || isa(grad_list[i], grad_T2s) || isa(grad_list[i], grad_B1)
            add_partial_derivative!(du_v, u_v1, x -> h(p, x; idxs=4), (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g, dG_o_dT2s_x_T2s), t, grad_list[i])
        end
    end
    return ∂m∂t
end

function apply_hamiltonian_gbloch_inversion!(∂m∂t, m, h, p, t)
    apply_hamiltonian_gbloch!(∂m∂t, m, h, p, t; pulsetype=:inversion)
end

###################################################
# Bloch-McConnel model to simulate free precession
###################################################
function apply_hamiltonian_freeprecession!(∂m∂t, m, p::NTuple{5,Any}, t)
    ω0, m0s, R1, R2f, Rx = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - (R1 + Rx * m0s) * m[3] + Rx * (1 - m0s)  * m[4] + (1 - m0s) * R1 * m[5]
    ∂m∂t[4] =   Rx * m0s  * m[3] - (R1 + Rx * (1 - m0s)) * m[4] + m0s  * R1 * m[5]
    return ∂m∂t
end

function apply_hamiltonian_freeprecession!(∂m∂t, m, p::NTuple{6,Any}, t)
    ω0, m0s, R1, R2f, Rx, grad_list = p

    # Apply Hamiltonian to M
    u_v1 = @view m[1:5]
    du_v1 = @view ∂m∂t[1:5]
    apply_hamiltonian_freeprecession!(du_v1, u_v1, (ω0, m0s, R1, R2f, Rx), t)

    # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view ∂m∂t[5 * i + 1:5 * (i + 1)]
        u_v  = @view m[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_freeprecession!(du_v, u_v, (ω0, m0s, R1, R2f, Rx), t)

        add_partial_derivative!(du_v, u_v1, [], (0.0, 1.0, ω0, m0s, R1, R2f, [], Rx, [], []), t, grad_list[i])
        # TODO: repalce [] with undef
    end
    return ∂m∂t
end

###################################################
# implementatoin of the partial derivates for 
# calculationg th gradient
###################################################
function add_partial_derivative!(∂m∂t, m, h, p, t, grad_type::grad_m0s)
    _, _, _, _, R1, _, _, Rx, _, _ = p

    ∂m∂t[3] -= Rx * m[3] + Rx * m[4] + R1
    ∂m∂t[4] += Rx * m[3] + Rx * m[4] + R1
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, h, p, t, grad_type::grad_R1)
    _, _, _, m0s, _, _, _, _, _, _ = p

    ∂m∂t[3] += - m[3] + (1 - m0s)
    ∂m∂t[4] += - m[4] + m0s
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, h, p, t, grad_type::grad_R2f)
    ∂m∂t[1] -= m[1]
    ∂m∂t[2] -= m[2]
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, h, p, t, grad_type::grad_Rx)
    _, _, _, m0s, _, _, _, _, _, _ = p

    ∂m∂t[3] += - m0s * m[3] + (1 - m0s) * m[4]
    ∂m∂t[4] +=   m0s * m[3] - (1 - m0s) * m[4]
    return ∂m∂t
end

# version for gBloch
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Fun,Fun}, t, grad_type::grad_T2s)
    ω1, B1, ω0, _, _, _, T2s, _, _, dG_o_dT2s_x_T2s = p


    yt = cos(ω0 * t) * quadgk(x -> dG_o_dT2s_x_T2s((t - x) / T2s) * h(x) * cos(ω0 * x), 0, t, order=100)[1]
    xt = sin(ω0 * t) * quadgk(x -> dG_o_dT2s_x_T2s((t - x) / T2s) * h(x) * sin(ω0 * x), 0, t, order=100)[1]
                      
    ∂m∂t[4] -= B1^2 * ω1^2 * (xt + yt)/T2s
    return ∂m∂t
end

# version for free precession (does nothing)
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Array{Any,1},Array{Any,1}}, t, grad_type::grad_T2s)
    return ∂m∂t
end

# version for Graham's model
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Number,Any}, t, grad_type::grad_T2s)
    ω1, B1, _, _, _, _, T2s, _, TRF, _ = p
    
    df_PSD = (τ) -> quadgk(ct -> 8 / τ * (exp(-τ^2 / 8 * (3 * ct^2 - 1)^2) - 1) / (3 * ct^2 - 1)^2 + sqrt(2π) * erf(τ / sqrt(8) * abs(3 * ct^2 - 1)) / abs(3 * ct^2 - 1), 0.0, 1.0, order=100)[1]
        
    ∂m∂t[4] -= df_PSD(TRF / T2s) * B1^2 * ω1^2 * m[4]
    return ∂m∂t
end

# version for gBloch model
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Fun,Fun}, t, grad_type::grad_ω0)
    ω1, B1, ω0, _, _, _, T2s, _, g, _ = p

    ∂m∂t[1] -= m[2]
    ∂m∂t[2] += m[1]

    yt = -sin(ω0 * t) * t * quadgk(x -> g((t - x) / T2s) * h(x) * cos(ω0 * x), 0, t, order=100)[1]
    yt += cos(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(x) * (-x) * sin(ω0 * x), 0, t, order=100)[1]

    xt = cos(ω0 * t) * t * quadgk(x -> g((t - x) / T2s) * h(x) * sin(ω0 * x), 0, t, order=100)[1]
    xt = sin(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(x) * x * cos(ω0 * x), 0, t, order=100)[1]

    ∂m∂t[4] -= B1^2 * ω1^2 * (xt + yt)
    return ∂m∂t
end

# version for free precession & Graham's model
function add_partial_derivative!(∂m∂t, m, h, p, t, grad_type::grad_ω0)
    ∂m∂t[1] -= m[2]
    ∂m∂t[2] += m[1]
    return ∂m∂t
end

# version for gBloch (using ApproxFun)
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Fun,Any}, t, grad_type::grad_B1)
    ω1, B1, ω0, _, _, _, T2s, _, g, _ = p
    
    ∂m∂t[1] += ω1 * m[3]
    ∂m∂t[3] -= ω1 * m[1]

    yt = cos(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(x) * cos(ω0 * x), 0, t, order=100)[1]
    xt = sin(ω0 * t) * quadgk(x -> g((t - x) / T2s) * h(x) * sin(ω0 * x), 0, t, order=100)[1]

    ∂m∂t[4] -= 2 * B1 * ω1^2 * (xt + yt)
    return ∂m∂t
end

# version for free precession (does nothing)
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Array{Any,1},Array{Any,1}}, t, grad_type::grad_B1)
    return ∂m∂t
end

# version for Graham
function add_partial_derivative!(∂m∂t, m, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Number,Any}, t, grad_type::grad_B1)
    ω1, B1, _, _, _, _, T2s, _, TRF, _ = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0, order=100)[1]

    ∂m∂t[1] += ω1 * m[3]
    ∂m∂t[3] -= ω1 * m[1]
    ∂m∂t[4] -= f_PSD(TRF / T2s) * 2 * B1 * ω1^2 * T2s * m[4]
    return ∂m∂t
end


###################################################
# Implementation for comparison: the super-Lorentzian 
# Green's function is hard coded, which allows to 
# use special solvers for the double integral
###################################################
function apply_hamiltonian_gbloch_superlorentzian!(∂m∂t, m, h, p::NTuple{10,Any}, t)
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
        
        ∂m∂t[4] = -B1^2 * ω1^2 * ((dy1) + (dy2))
    else
        ∂m∂t[4] = -B1^2 * ω1^2 * (dy1)
    end

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1 * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1  * m[1] - (R1 + Rx * m0s) * m[3] +       Rx * (1 - m0s)  * m[4] + (1 - m0s) * R1 * m[5]
    ∂m∂t[4] +=             +       Rx * m0s  * m[3] - (R1 + Rx * (1 - m0s)) * m[4] +      m0s  * R1 * m[5]
    ∂m∂t[5] = 0.0
    return ∂m∂t
end

function apply_hamiltonian_gbloch_superlorentzian!(∂m∂t, m, h, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, N = p
    return apply_hamiltonian_gbloch_superlorentzian!(∂m∂t, m, h, (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, 4, N), t)
end

###################################################
# Graham's spectral model
###################################################
function apply_hamiltonian_graham_superlorentzian!(∂m∂t, m, p::NTuple{9,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0, order=100)[1]
    Rrf = f_PSD(TRF / T2s) * B1^2 * ω1^2 * T2s

    return apply_hamiltonian_linear!(∂m∂t, m, (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf), t)
end

function apply_hamiltonian_graham_superlorentzian!(∂m∂t, m, p::NTuple{10,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx, grad_list = p
    
    # Apply Hamiltonian to M
    u_v1 = @view m[1:5]
    du_v1 = @view ∂m∂t[1:5]
    apply_hamiltonian_graham_superlorentzian!(du_v1, u_v1, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
    # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view ∂m∂t[5 * i + 1:5 * (i + 1)]
        u_v  = @view m[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_graham_superlorentzian!(du_v, u_v, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
        add_partial_derivative!(du_v, u_v1, [], (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, TRF, []), t, grad_list[i])
    end
    return ∂m∂t
end

function Graham_Hamiltonian_superLorentzian_InversionPulse!(∂m∂t, m, p::NTuple{10,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx, grad_list = p
    
    # Apply Hamiltonian to M
    u_v1 = @view m[1:5]
    du_v1 = @view ∂m∂t[1:5]
    apply_hamiltonian_graham_superlorentzian!(du_v1, u_v1, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
    # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view ∂m∂t[5 * i + 1:5 * (i + 1)]
        u_v  = @view m[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_graham_superlorentzian!(du_v, u_v, (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx), t)
 
        if isa(grad_list[i], grad_B1) || isa(grad_list[i], grad_T2s)
            add_partial_derivative!(du_v, u_v1, [], (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, TRF, []), t, grad_list[i])
        end
    end
    return ∂m∂t
end


function apply_hamiltonian_linear!(∂m∂t, m, p::NTuple{8,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf = p
    
    apply_hamiltonian_freeprecession!(∂m∂t, m, (ω0, m0s, R1, R2f, Rx), t)

    ∂m∂t[1] += B1 * ω1 * m[3]
    ∂m∂t[3] -= B1 * ω1 * m[1]
    ∂m∂t[4] -= Rrf * m[4]
    return ∂m∂t
end

function apply_hamiltonian_linear!(∂m∂t, m, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf_d, grad_list = p
    Rrf = Rrf_d[1]
    
     # Apply Hamiltonian to M
    u_v1 = @view m[1:5]
    du_v1 = @view ∂m∂t[1:5]
    apply_hamiltonian_linear!(du_v1, u_v1, (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf), t)
 
     # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view ∂m∂t[5 * i + 1:5 * (i + 1)]
        u_v  = @view m[5 * i + 1:5 * (i + 1)]
        apply_hamiltonian_linear!(du_v, u_v, (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf), t)
 
        add_partial_derivative!(du_v, u_v1, [], (ω1, B1, ω0, m0s, R1, R2f, 0.0, Rx, Rrf_d, []), t, grad_list[i])
    end
    return ∂m∂t
end

##################################################################
# Sled's model
##################################################################
"""
    apply_hamiltonian_sled!(∂m∂t, m, p, t)

Apply Sled's Hamiltonian to `m` and write the resulting derivative wrt. time into `∂m∂t`.

Currently, this funciton is only implemented for an isolated semi-solid spin pool.

# Arguemnts
- `∂m∂t::Vector{<:Number}`: Vector of length 1 describing to derivative of `m` wrt. time; this vector can contain any value, which is replaced by `H * m`
- `m::Vector{<:Number}`: Vector of length 1 describing the `zs` magnetization
- `p::NTuple{9,10, or 11, Any}`: `(ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, g)`, with 
    -`ω1::Number`: Rabi frequency in rad/s (rotation about the y-axis)
    -`B1::Number`: B1 scaling normalized so that `B1=1` corresponds to a perfectly calibrated RF field
    -`ω0::Number`: Larmor or off-resonance frequency in rad/s
    -`R1::Number`: Longitudinal spin relaxation rate in 1/seconds
    -`T2s::Number`: Trasversal spin relaxation time in seconds
    -`g::Function`: Green's function of the form `G(κ) = G((t-τ)/T2s)`
- `t::Number`: Time in seconds

# Examples
```jldoctest
julia> using DifferentialEquations

julia> α = π/2;

julia> TRF = 100e-6;

julia> ω1 = α/TRF;

julia> B1 = 1;

julia> ω0 = 0;

julia> R1 = 1;

julia> T2s = 10e-6;

julia> G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

julia> m0 = [1];

julia> sol = solve(ODEProblem(apply_hamiltonian_sled!, m0, (0, TRF), (ω1, 1, ω0, R1, T2s, G)), Tsit5())
retcode: Success
Interpolation: specialized 4th order "free" interpolation
t: 3-element Vector{Float64}:
 0.0
 7.475658194333419e-5
 0.0001
u: 3-element Vector{Vector{Float64}}:
 [1.0]
 [0.6313685535188782]
 [0.4895191983659201]

julia> using Plots

julia> plot(sol, labels=["zs"], xlabel="t (s)", ylabel="m(t)");

```
"""
function apply_hamiltonian_sled!(d∂m∂t, m, p::NTuple{6,Any}, t)
    ω1, B1, ω0, R1, T2s, g = p

    yt = cos(ω0 * t) * quadgk(x -> g((t - x) / T2s) * cos(ω0 * x), 0, t, order=100)[1]
    xt = sin(ω0 * t) * quadgk(x -> g((t - x) / T2s) * sin(ω0 * x), 0, t, order=100)[1]
    
    d∂m∂t[1] = -B1^2 * ω1^2 * (xt + yt) * m[1] + R1 * (1 - m[1])
end
