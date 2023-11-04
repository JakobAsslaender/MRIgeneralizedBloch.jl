###################################################
# generalized Bloch Hamiltonians that can take any
# Green's function as an argument.
###################################################
"""
    apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p, t)

Apply the generalized Bloch Hamiltonian to `m` and write the resulting derivative wrt. time into `∂m∂t`.

# Arguments
- `∂m∂t::Vector{Real}`: Vector describing to derivative of `m` wrt. time; this vector has to be of the same size as `m`, but can contain any value, which is replaced by `H * m`
- `m::Vector{Real}`: Vector the spin ensemble state of the form `[xf, yf, zf, zs, 1]` if now gradient is calculated or of the form `[xf, yf, zf, zs, 1, ∂xf/∂θ1, ∂yf/∂θ1, ∂zf/∂θ1, ∂zs/∂θ1, 0, ..., ∂xf/∂θn, ∂yf/∂θn, ∂zf/∂θn, ∂zs/∂θn, 0]` if n derivatives wrt. `θn` are calculated
- `mfun`: History function; can be initialized with `mfun(p, t; idxs=nothing) = typeof(idxs) <: Real ? 0.0 : zeros(5n + 5)` for n gradients, and is then updated by the delay differential equation solvers
- `p::NTuple{6,Any}`: `(ω1, B1, ω0, R1s, T2s, g)` or
- `p::NTuple{6,Any}`: `(ω1, B1,  φ, R1s, T2s, g)` or
- `p::NTuple{10,Any}`: `(ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g)` or
- `p::NTuple{10,Any}`: `(ω1, B1,  φ, m0s, R1f, R2f, Rx, R1s, T2s, g)` or
- `p::NTuple{12,Any}`: `(ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s, grad_list)` or
- `p::NTuple{12,Any}`: `(ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s, grad_list)` with the following entries
    - `ω1::Real`: Rabi frequency in rad/s (rotation about the y-axis) or
    - `ω1(t)::Function`: Rabi frequency in rad/s as a function of time for shaped RF-pulses
    - `B1::Real`: B1 scaling normalized so that `B1=1` corresponds to a perfectly calibrated RF field
    - `ω0::Real`: Larmor or off-resonance frequency in rad/s or
    - `φ::Function`: RF-phase in rad as a function of time for frequency/phase-sweep pulses (works only in combination with `ω1(t)::Function`)
    - `m0s::Real`: Fractional semi-solid spin pool size in the range of 0 to 1
    - `R1f::Real`: Longitudinal spin relaxation rate of the free pool in 1/seconds
    - `R2f::Real`: Transversal spin relaxation rate of the free pool in 1/seconds
    - `Rx::Real`: Exchange rate between the two pools in 1/seconds
    - `R1s::Real`: Longitudinal spin relaxation rate of the semi-solid pool in 1/seconds
    - `T2s::Real`: Transversal spin relaxation time of the semi-solid pool in seconds
    - `g::Function`: Green's function of the form `G(κ) = G((t-τ)/T2s)`
    - `dG_o_dT2s_x_T2s::Function`: Derivative of the Green's function wrt. T2s, multiplied by T2s; of the form `dG_o_dT2s_x_T2s(κ) = dG_o_dT2s_x_T2s((t-τ)/T2s)`
    - `grad_list::Vector{grad_param}`: List of gradients to be calculated, i.e., any subset of `[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]`; length of the vector must be n (cf. arguments `m` and `∂m∂t`); the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `t::Real`: Time in seconds

Optional:
- `pulsetype=:normal`: Use default for a regular RF-pulse; the option `pulsetype=:inversion` should be handled with care as it is only intended to calculate the saturation of the semi-solid pool and its derivative.

# Examples
```jldoctest
julia> using DifferentialEquations


julia> α = π/2;

julia> TRF = 100e-6;

julia> ω1 = α/TRF;

julia> B1 = 1;

julia> ω0 = 0;

julia> m0s = 0.2;

julia> R1f = 1/3;

julia> R2f = 15;

julia> R1s = 2;

julia> T2s = 10e-6;

julia> Rx = 30;

julia> G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);


julia> m0 = [0; 0; 1-m0s; m0s; 1];

julia> mfun(p, t; idxs=nothing) = typeof(idxs) <: Real ? 0.0 : zeros(5);

julia> sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)))
retcode: Success
Interpolation: specialized 4th order "free" interpolation, specialized 2nd order "free" stiffness-aware interpolation
t: 9-element Vector{Float64}:
 0.0
 1.375006182301112e-7
 1.512506800531223e-6
 8.042561696923577e-6
 2.107848894861101e-5
 3.9114182159866e-5
 6.26879358261189e-5
 9.147711414688425e-5
 0.0001
u: 9-element Vector{Vector{Float64}}:
 [0.0, 0.0, 0.8, 0.2, 1.0]
 [0.0017278806030763402, 0.0, 0.7999981340131751, 0.19999953350448, 1.0]
 [0.019004717382235078, 0.0, 0.7997742277135814, 0.19994357804868362, 1.0]
 [0.10079111348917136, 0.0, 0.7936248122939504, 0.19842287240368398, 1.0]
 [0.2600257867257624, 0.0, 0.7565529666157949, 0.1898191304278861, 1.0]
 [0.46104237829774064, 0.0, 0.6537239462232086, 0.16937683398576228, 1.0]
 [0.6661740376622253, 0.0, 0.44261209248221817, 0.13589311206074786, 1.0]
 [0.7923117772809817, 0.0, 0.10713073823030607, 0.09390260581965477, 1.0]
 [0.7994211188442756, 0.0, 0.0004403374305009168, 0.08214809659226184, 1.0]

julia> using Plots

julia> plot(sol, labels=["xf" "yf" "zf" "zs" "1"], xlabel="t (s)", ylabel="m(t)");




julia> dG_o_dT2s_x_T2s = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, TRF / T2s);


julia> grad_list = [grad_R2f(), grad_m0s()];


julia> m0 = [0; 0; 1-m0s; m0s; 1; zeros(5*length(grad_list))];


julia> mfun(p, t; idxs=nothing) = typeof(idxs) <: Real ? 0.0 : zeros(5 + 5*length(grad_list));

julia> sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G, dG_o_dT2s_x_T2s, grad_list)));




julia> plot(sol);


```
"""
function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::NTuple{11,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, zs_idx, g = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1 * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1  * m[1] - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1f * m[5]

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> exp(1im * ω0 * τ) * g((t - τ) / T2s) * mfun(p, τ; idxs=zs_idx), eps(), t, order=7)[1])
    ∂m∂t[4] = -B1^2 * ω1^2 * xys + Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s * R1s * m[5]
    return ∂m∂t
end

function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::Tuple{Function,Real,Real,Real,Real,Real,Real,Real,Real,Integer,Function}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, zs_idx, g = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1(t) * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1(t)  * m[1] - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1f * m[5]

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> ω1(τ) * exp(1im * ω0 * τ) * g((t - τ) / T2s) * mfun(p, τ; idxs=zs_idx), eps(), t, order=7)[1])
    ∂m∂t[4] = -B1^2 * ω1(t) * xys + Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s * R1s * m[5]
    return ∂m∂t
end

function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::Tuple{Function,Real,Function,Real,Real,Real,Real,Real,Real,Integer,Function}, t)
    ω1, B1, φ, m0s, R1f, R2f, Rx, R1s, T2s, zs_idx, g = p

    ∂m∂t[1] = - R2f * m[1] + B1 * ω1(t) * cos(φ(t)) * m[3]
    ∂m∂t[2] = - R2f * m[2] - B1 * ω1(t) * sin(φ(t)) * m[3]
    ∂m∂t[3] = - B1 * ω1(t) * cos(φ(t)) * m[1] + B1 * ω1(t) * sin(φ(t)) * m[2] - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1f * m[5]

    xys = real(exp(-1im * φ(t)) * quadgk(τ -> ω1(τ) * exp(1im * φ(τ)) * g((t - τ) / T2s) * mfun(p, τ; idxs=zs_idx), eps(), t, order=7)[1])
    ∂m∂t[4] = -B1^2 * ω1(t) * xys + Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s * R1s * m[5]
    return ∂m∂t
end

function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::NTuple{10,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g = p
    return apply_hamiltonian_gbloch!(∂m∂t, m, mfun, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, 4, g), t)
end

# Version for an isolated semi-solid pool
function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::NTuple{6,Any}, t)
    ω1, B1, ω0, R1s, T2s, g = p

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> exp(1im * ω0 * τ) * g((t - τ) / T2s) * mfun(p, τ)[1], 0, t, order=7)[1])
    ∂m∂t[1] = -B1^2 * ω1^2 * xys + R1s * (1 - m[1])
    return ∂m∂t
end

function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::Tuple{Function,Real,Real,Real,Real,Function}, t)
    ω1, B1, ω0, R1s, T2s, g = p

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> ω1(τ) * exp(1im * ω0 * τ) * g((t - τ) / T2s) * mfun(p, τ)[1], 0, t, order=7)[1])
    ∂m∂t[1] = -B1^2 * ω1(t) * xys + R1s * (1 - m[1])
    return ∂m∂t
end

function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::Tuple{Function,Real,Function,Real,Real,Function}, t)
    ω1, B1, φ, R1s, T2s, g = p

    xys = real(exp(-1im * φ(t)) * quadgk(τ -> ω1(τ) * exp(1im * φ(τ)) * g((t - τ) / T2s) * mfun(p, τ)[1], 0, t, order=7)[1])
    ∂m∂t[1] = -B1^2 * ω1(t) * xys + R1s * (1 - m[1])
    return ∂m∂t
end


function apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p::NTuple{12,Any}, t; pulsetype=:normal)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s, grad_list = p

    ∂m∂t_m = reshape(∂m∂t, 5, :)
    m_m    = reshape(   m, 5, :)
    mfun4(τ) = mfun(p, τ; idxs=4)

    # Apply Hamiltonian to M, all derivatives and add partial derivatives
    for i ∈ axes(m_m, 2)
        @views apply_hamiltonian_gbloch!(∂m∂t_m[:,i], m_m[:,i], mfun, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, (5i - 1), g), t)

        if i > 1 && (pulsetype==:normal || isa(grad_list[i-1], grad_T2s) || isa(grad_list[i-1], grad_B1))
            # @views add_partial_derivative!(∂m∂t_m[:,i], m_m[:,1], τ -> mfun(p, τ; idxs=4), (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s), t, grad_list[i-1])
            @views add_partial_derivative!(∂m∂t_m[:,i], m_m[:,1], mfun4, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s), t, grad_list[i-1])
        end
    end
    return ∂m∂t
end

function apply_hamiltonian_gbloch_inversion!(∂m∂t, m, mfun, p, t)
    apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p, t; pulsetype=:inversion)
end

###################################################
# Bloch-McConnell model to simulate free precession
###################################################
function apply_hamiltonian_freeprecession!(∂m∂t, m, p::NTuple{6,Any}, t)
    ω0, m0s, R1f, R2f, Rx, R1s = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s)  * m[4] + (1 - m0s) * R1f * m[5]
    ∂m∂t[4] =   Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s  * R1s * m[5]
    return ∂m∂t
end

function apply_hamiltonian_freeprecession!(∂m∂t, m, p::NTuple{7,Any}, t)
    ω0, m0s, R1f, R2f, Rx, R1s, grad_list = p

    ∂m∂t_m = reshape(∂m∂t, 5, :)
    m_m    = reshape(   m, 5, :)

    # Apply Hamiltonian to M, all derivatives and add partial derivatives
    for i ∈ axes(m_m, 2)
        @views apply_hamiltonian_freeprecession!(∂m∂t_m[:,i], m_m[:,i], (ω0, m0s, R1f, R2f, Rx, R1s), t)

        if i > 1
            @views add_partial_derivative!(∂m∂t_m[:,i], m_m[:,1], undef, (0, 1, ω0, m0s, R1f, R2f, Rx, R1s, undef, undef, undef), t, grad_list[i-1])
        end
    end
    return ∂m∂t
end

#########################################################################
# implementation of the partial derivatives for calculating the gradients
#########################################################################
function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_m0s)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, _, dG_o_dT2s_x_T2s = p

    ∂m∂t[3] -= Rx * m[3] + Rx * m[4] + R1f * m[5]
    ∂m∂t[4] += Rx * m[3] + Rx * m[4] + R1s * m[5]
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_R1a)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, _, dG_o_dT2s_x_T2s = p

    ∂m∂t[3] += - m[3] + (1 - m0s)
    ∂m∂t[4] += - m[4] + m0s
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_R1f)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, _, dG_o_dT2s_x_T2s = p

    ∂m∂t[3] += - m[3] + (1 - m0s)
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_R1s)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, _, dG_o_dT2s_x_T2s = p

    ∂m∂t[4] += - m[4] + m0s
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_R2f)
    ∂m∂t[1] -= m[1]
    ∂m∂t[2] -= m[2]
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_Rx)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, TRF, dG_o_dT2s_x_T2s = p

    ∂m∂t[3] += - m0s * m[3] + (1 - m0s) * m[4]
    ∂m∂t[4] +=   m0s * m[3] - (1 - m0s) * m[4]
    return ∂m∂t
end

# versions for gBloch
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Real,Real,Real,Any,Any,Any,Any,Any,Real,Function,Function}, t, grad_type::grad_T2s)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> exp(1im * ω0 * τ) * dG_o_dT2s_x_T2s((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= B1^2 * ω1^2 * xys/T2s
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Function,Real,Real,Any,Any,Any,Any,Any,Real,Function,Function}, t, grad_type::grad_T2s)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> ω1(τ) * exp(1im * ω0 * τ) * dG_o_dT2s_x_T2s((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= B1^2 * ω1(t) * xys/T2s
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Function,Real,Function,Any,Any,Any,Any,Any,Real,Function,Function}, t, grad_type::grad_T2s)
    ω1, B1, φ, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    xys = real(exp(-1im * φ(t)) * quadgk(τ -> ω1(τ) * exp(1im * φ(τ)) * dG_o_dT2s_x_T2s((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= B1^2 * ω1(t) * xys/T2s
    return ∂m∂t
end

# version for free precession (does nothing)
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any,UndefInitializer,UndefInitializer}, t, grad_type::grad_T2s)
    return ∂m∂t
end

# versions for Graham's model
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Real,Real,Real,Real,Real,Real,Real,Real,Real,Real,Real}, t, grad_type::grad_T2s)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, Rrf, dRrfdT2s = p

    ∂m∂t[4] -= dRrfdT2s * m[4]
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Real,Any,Any,Any,Any,Any,Any,Any,Real,Real,Any}, t, grad_type::grad_T2s)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, TRF, dG_o_dT2s_x_T2s = p

    df_PSD(τ) = quadgk(ct -> 8 / τ * (exp(-τ^2 / 8 * (3 * ct^2 - 1)^2) - 1) / (3 * ct^2 - 1)^2 + sqrt(2π) * erf(τ / sqrt(8) * abs(3 * ct^2 - 1)) / abs(3 * ct^2 - 1), 0.0, 1.0, order=7)[1]

    ∂m∂t[4] -= df_PSD(TRF / T2s) * B1^2 * ω1^2 * m[4]
    return ∂m∂t
end

# versions for gBloch model
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Real,Real,Real,Any,Any,Any,Any,Any,Real,Function,Function}, t, grad_type::grad_ω0)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    ∂m∂t[1] -= m[2]
    ∂m∂t[2] += m[1]

    xys  = imag(exp(-1im * ω0 * t) * t * quadgk(τ -> exp(1im * ω0 * τ)     * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    xys -= imag(exp(-1im * ω0 * t)     * quadgk(τ -> exp(1im * ω0 * τ) * τ * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= B1^2 * ω1^2 * xys
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Function,Real,Real,Any,Any,Any,Any,Any,Real,Function,Function}, t, grad_type::grad_ω0)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    ∂m∂t[1] -= m[2]
    ∂m∂t[2] += m[1]

    xys  = imag(exp(-1im * ω0 * t) * t * quadgk(τ -> ω1(τ) * exp(1im * ω0 * τ)     * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    xys -= imag(exp(-1im * ω0 * t)     * quadgk(τ -> ω1(τ) * exp(1im * ω0 * τ) * τ * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= B1^2 * ω1(t) * xys
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Function,Real,Function,Any,Any,Any,Any,Any,Real,Function,Function}, t, grad_type::grad_ω0)
    ω1, B1, φ, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    ∂m∂t[1] -= B1 * ω1(t) * sin(φ(t)) * t * m[3]
    ∂m∂t[2] -= B1 * ω1(t) * cos(φ(t)) * t * m[3]
    ∂m∂t[3] += B1 * ω1(t) * sin(φ(t)) * t * m[1] + B1 * ω1(t) * cos(φ(t)) * t * m[2]

    xys  = imag(exp(-1im * φ(t)) * t * quadgk(τ -> ω1(τ) * exp(1im * φ(τ))     * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    xys -= imag(exp(-1im * φ(t))     * quadgk(τ -> ω1(τ) * exp(1im * φ(τ)) * τ * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= B1^2 * ω1(t) * xys
    return ∂m∂t
end

# version for free precession & Graham's model
function add_partial_derivative!(∂m∂t, m, mfun, p::NTuple{11,Any}, t, grad_type::grad_ω0)
    ∂m∂t[1] -= m[2]
    ∂m∂t[2] += m[1]
    return ∂m∂t
end

# versions for gBloch (using ApproxFun)
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Real,Real,Real,Any,Any,Any,Any,Any,Real,Function,Any}, t, grad_type::grad_B1)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    ∂m∂t[1] += ω1 * m[3]
    ∂m∂t[3] -= ω1 * m[1]

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> exp(1im * ω0 * τ) * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= 2 * B1 * ω1^2 * xys
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Function,Real,Real,Any,Any,Any,Any,Any,Real,Function,Any}, t, grad_type::grad_B1)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p

    ∂m∂t[1] += ω1(t) * m[3]
    ∂m∂t[3] -= ω1(t) * m[1]

    xys = real(exp(-1im * ω0 * t) * quadgk(τ -> ω1(τ) * exp(1im * ω0 * τ) * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])
    ∂m∂t[4] -= 2 * B1 * ω1(t) * xys
    return ∂m∂t
end

function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Function,Real,Function,Any,Any,Any,Any,Any,Real,Function,Any}, t, grad_type::grad_B1)
    ω1, B1, φ, m0s, R1f, R2f, Rx, R1s, T2s, g, dG_o_dT2s_x_T2s = p


    ∂m∂t[1] += ω1(t) * cos(φ(t)) * m[3]
    ∂m∂t[2] -= ω1(t) * sin(φ(t)) * m[3]
    ∂m∂t[3] += - ω1(t) * cos(φ(t)) * m[1] + ω1(t) * sin(φ(t)) * m[2]

    xys = real(exp(-1im * φ(t)) * quadgk(τ -> ω1(τ) * exp(1im * φ(τ)) * g((t - τ) / T2s) * mfun(τ), 0, t, order=7)[1])

    ∂m∂t[4] -= 2 * B1 * ω1(t) * xys
    return ∂m∂t
end

# version for free precession (does nothing)
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any,UndefInitializer,UndefInitializer}, t, grad_type::grad_B1)
    return ∂m∂t
end

# version for Graham
function add_partial_derivative!(∂m∂t, m, mfun, p::Tuple{Real,Real,Any,Any,Any,Any,Any,Any,Real,Real,Any}, t, grad_type::grad_B1)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, TRF, dG_o_dT2s_x_T2s = p

    f_PSD(τ) = quadgk(ct -> 1 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0, 1, order=7)[1]

    ∂m∂t[1] += ω1 * m[3]
    ∂m∂t[3] -= ω1 * m[1]
    ∂m∂t[4] -= f_PSD(TRF / T2s) * 2 * B1 * ω1^2 * T2s * m[4]
    return ∂m∂t
end

##############################################################################
# Implementation for comparison: the super-Lorentzian Green's function
# is hard coded, which allows to use special solvers for the double integral
##############################################################################
function apply_hamiltonian_gbloch_superlorentzian!(∂m∂t, m, mfun, p::NTuple{11,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, zs_idx, N = p

    gt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3 * ct^2 - 1)^2 / 8)

    function fy!(x, y, gt, mfun, p, T2s, zs_idx, t)
        for i = 1:size(x, 2)
            y[i] = gt(t - x[2,i], T2s, x[1,i]) * mfun(p, x[2,i]; idxs=zs_idx)
        end
    end

    dy1 = Cubature.pcubature_v((x, y) -> fy!(x, y, gt, mfun, p, T2s, zs_idx, t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = T2s * sqrt(2π / 3) * Cubature.pcubature(x -> mfun(p, x[1]; idxs=zs_idx) / (t - x[1]), [0.0], [t - N * T2s])[1]

        ∂m∂t[4] = -B1^2 * ω1^2 * ((dy1) + (dy2))
    else
        ∂m∂t[4] = -B1^2 * ω1^2 * (dy1)
    end

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1 * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1  * m[1] - (R1f + Rx * m0s) * m[3] +       Rx * (1 - m0s)  * m[4] + (1 - m0s) * R1f * m[5]
    ∂m∂t[4] +=             +       Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] +      m0s  * R1s * m[5]
    ∂m∂t[5] = 0.0
    return ∂m∂t
end

function apply_hamiltonian_gbloch_superlorentzian!(∂m∂t, m, mfun, p::NTuple{10,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, N = p
    return apply_hamiltonian_gbloch_superlorentzian!(∂m∂t, m, mfun, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, 4, N), t)
end



###################################################
# Graham's spectral model
###################################################
function apply_hamiltonian_graham_superlorentzian!(∂m∂t, m, p::NTuple{10,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s = p

    f_PSD(τ) = quadgk(ct -> 1 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0, 1, order=7)[1]
    Rrf = f_PSD(TRF / T2s) * B1^2 * ω1^2 * T2s

    return apply_hamiltonian_linear!(∂m∂t, m, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf), t)
end

function apply_hamiltonian_graham_superlorentzian!(∂m∂t, m, p::NTuple{11,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, grad_list = p

    ∂m∂t_m = reshape(∂m∂t, 5, :)
    m_m    = reshape(   m, 5, :)

    # Apply Hamiltonian to M, all derivatives and add partial derivatives
    for i ∈ axes(m_m, 2)
        @views apply_hamiltonian_graham_superlorentzian!(∂m∂t_m[:,i], m_m[:,i], (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s), t)

        if i > 1
            @views add_partial_derivative!(∂m∂t_m[:,i], m_m[:,1], undef, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, TRF, undef), t, grad_list[i-1])
        end
    end
    return ∂m∂t
end

function apply_hamiltonian_graham_superlorentzian_inversionpulse!(∂m∂t, m, p::NTuple{11,Any}, t)
    ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s, grad_list = p

    ∂m∂t_m = reshape(∂m∂t, 5, :)
    m_m    = reshape(   m, 5, :)

    # Apply Hamiltonian to M, all derivatives and add partial derivatives
    for i ∈ axes(m_m, 2)
        @views apply_hamiltonian_graham_superlorentzian!(∂m∂t_m[:,i], m_m[:,i], (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rx, R1s, T2s), t)

        if i > 1 && (isa(grad_list[i-1], grad_B1) || isa(grad_list[i-1], grad_T2s))
            @views add_partial_derivative!(∂m∂t_m[:,i], m_m[:,1], undef, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, TRF, undef), t, grad_list[i-1])
        end
    end
    return ∂m∂t
end

function apply_hamiltonian_linear!(∂m∂t, m, p::Tuple{Function,Real,Real,Real,Real,Real,Real,Real,Real}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf = p
    apply_hamiltonian_linear!(∂m∂t, m, (ω1(t), B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf), t)
end
function apply_hamiltonian_linear!(∂m∂t, m, p::Tuple{Function,Real,Real,Real,Real,Real,Real,Real,Real,Real,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf, dRrfdT2s, grad_list = p
    return apply_hamiltonian_linear!(∂m∂t, m, (ω1(t), B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf, dRrfdT2s, grad_list), t)
end

function apply_hamiltonian_linear!(∂m∂t, m, p::Tuple{Function,Real,Function,Real,Real,Real,Real,Real,Real}, t)
    ω1, B1, φ, m0s, R1f, R2f, Rx, R1s, Rrf = p

    apply_hamiltonian_freeprecession!(∂m∂t, m, (0, m0s, R1f, R2f, Rx, R1s), t)

    ∂m∂t[1] += B1 * ω1(t) * cos(φ(t)) * m[3]
    ∂m∂t[2] -= B1 * ω1(t) * sin(φ(t)) * m[3]
    ∂m∂t[3] -= B1 * ω1(t) * cos(φ(t)) * m[1]
    ∂m∂t[3] += B1 * ω1(t) * sin(φ(t)) * m[2]
    ∂m∂t[4] -= Rrf * m[4]
    return ∂m∂t
end

function apply_hamiltonian_linear!(∂m∂t, m, p::NTuple{9,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf = p

    apply_hamiltonian_freeprecession!(∂m∂t, m, (ω0, m0s, R1f, R2f, Rx, R1s), t)

    ∂m∂t[1] += B1 * ω1 * m[3]
    ∂m∂t[3] -= B1 * ω1 * m[1]
    ∂m∂t[4] -= Rrf * m[4]
    return ∂m∂t
end

function apply_hamiltonian_linear!(∂m∂t, m, p::NTuple{11,Any}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf, dRrfdT2s, grad_list = p

    ∂m∂t_m = reshape(∂m∂t, 5, :)
    m_m    = reshape(   m, 5, :)

    # Apply Hamiltonian to M, all derivatives and add partial derivatives
    for i ∈ axes(m_m, 2)
        @views apply_hamiltonian_linear!(∂m∂t_m[:,i], m_m[:,i], (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, Rrf), t)

        if i > 1
            @views add_partial_derivative!(∂m∂t_m[:,i], m_m[:,1], undef, (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, 0, Rrf, dRrfdT2s), t, grad_list[i-1])
        end
    end
    return ∂m∂t
end




"""
    graham_saturation_rate_spectral(lineshape, ω1, TRF, Δω)

Calculate saturation rate (in units of 1/s) according to Graham's spectral model.

# Arguments
- `lineshape::Function`: as a function of ω₀ (in rad/s). Supply, e.g., the anonymous function `ω₀ -> lineshape_superlorentzian(ω₀, T2s)`. Note that the integral over the lineshape has to be 1.
- `ω1::Function`: ω1 in rad/s as a function of time (in units of s) where the puls shape is defined for t ∈ [0,TRF]
- `TRF::Real`: duration of the RF pulse in s
- `Δω::Real`: offset frequency in rad/s

# Examples
```jldoctest
julia> using SpecialFunctions

julia> T2s = 10e-6;

julia> α = π;

julia> TRF = 100e-6;

julia> NSideLobes = 1;

julia> ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1));

julia> Δω = 200;

julia> graham_saturation_rate_spectral(ω₀ -> lineshape_superlorentzian(ω₀, T2s), ω1, TRF, Δω)
56135.388046022905
```
"""
function graham_saturation_rate_spectral(lineshape::Function, ω1::Real, TRF::Real, Δω::Real)
    S(ω, Δω) = abs((exp(1im * TRF * (-Δω + ω)) - 1) * ω1 / (Δω - ω))^2 / (2π*TRF)
    Rrf = π * quadgk(ω -> S(ω, Δω) * lineshape(ω), -Inf, 0, Inf)[1]
    return Rrf
end
function graham_saturation_rate_spectral(lineshape::Function, ω1::Function, TRF::Real, Δω::Real)
    S(ω, Δω) = abs(quadgk(t -> ω1(t) * exp(1im * (ω - Δω) * t), 0, TRF)[1])^2 / (2π*TRF)
    Rrf = π * quadgk(ω -> S(ω, Δω) * lineshape(ω), -Inf, 0, Inf)[1]
    return Rrf
end
function graham_saturation_rate_spectral(lineshape::Function, ω1::Function, TRF::Real, φ::Function)
    S(ω, φ) = abs(quadgk(t -> ω1(t) * exp(1im * (ω * t + φ(t))), 0, TRF)[1])^2 / (2π*TRF)
    Rrf = π * quadgk(ω -> S(ω, φ) * lineshape(ω), -Inf, 0, Inf)[1]
    return Rrf
end


"""
    graham_saturation_rate_single_frequency(lineshape, ω1, TRF, Δω)

Calculate saturation rate (in units of 1/s) according to Graham's single frequency approximation.

# Arguments
- `lineshape::Function`: as a function of ω₀ (in rad/s). Supply, e.g., the anonymous function `ω₀ -> lineshape_superlorentzian(ω₀, T2s)`. Note that the integral over the lineshape has to be 1.
- `ω1::Function`: ω1 in rad/s as a function of time (in units of s) where the puls shape is defined for t ∈ [0,TRF]
- `TRF::Real`: duration of the RF pulse in s
- `Δω::Real`: offset frequency in rad/s

# Examples
```jldoctest
julia> using SpecialFunctions

julia> T2s = 10e-6;

julia> α = π;

julia> TRF = 100e-6;

julia> NSideLobes = 1;

julia> ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1));

julia> Δω = 200;

julia> graham_saturation_rate_single_frequency(ω₀ -> lineshape_superlorentzian(ω₀, T2s), ω1, TRF, Δω)
419969.3376658947
```
"""
function graham_saturation_rate_single_frequency(lineshape::Function, ω1::Function, TRF::Real, Δω::Real)
    p = quadgk(t -> ω1(t)^2, 0, TRF)[1] / TRF
    Rrf = π * p * lineshape(Δω)
    return Rrf
end
function graham_saturation_rate_single_frequency(lineshape::Function, ω1::Real, TRF::Real, Δω::Real)
    return graham_saturation_rate_single_frequency(lineshape, (t) -> ω1, TRF, Δω)
end


##################################################################
# Sled's model
##################################################################
"""
    apply_hamiltonian_sled!(∂m∂t, m, p, t)

Apply Sled's Hamiltonian to `m` and write the resulting derivative wrt. time into `∂m∂t`.

# Arguments
- `∂m∂t::Vector{<:Real}`: Vector of length 1 describing to derivative of `m` wrt. time; this vector can contain any value, which is replaced by `H * m`
- `m::Vector{<:Real}`: Vector of length 1 describing the `zs` magnetization
- `p::NTuple{6 or 10, Any}`: `(ω1, B1, ω0, R1s, T2s, g)` for a simulating an isolated semi-solid pool or `(ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g)` for simulating a coupled spin system; with
    - `ω1::Real`: Rabi frequency in rad/s (rotation about the y-axis) or
    - `ω1(t)::Function`: Rabi frequency in rad/s as a function of time for shaped RF-pulses
    - `B1::Real`: B1 scaling normalized so that `B1=1` corresponds to a perfectly calibrated RF field
    - `ω0::Real`: Larmor or off-resonance frequency in rad/s (is only used for the free spin pool)
    - `R1f::Real`: Longitudinal spin relaxation rate of the free pool in 1/seconds
    - `R2f::Real`: Transversal spin relaxation rate of the free pool in 1/seconds
    - `R1s::Real`: Longitudinal spin relaxation rate of the semi-solid in 1/seconds
    - `Rx::Real`: Exchange rate between the two pools in 1/seconds
    - `T2s::Real`: Transversal spin relaxation time in seconds
    - `g::Function`: Green's function of the form `G(κ) = G((t-τ)/T2s)`
- `t::Real`: Time in seconds

# Examples
```jldoctest
julia> using DifferentialEquations


julia> α = π/2;

julia> TRF = 100e-6;

julia> ω1 = α/TRF;

julia> B1 = 1;

julia> ω0 = 0;

julia> R1s = 2;

julia> T2s = 10e-6;

julia> G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

julia> m0 = [1];

julia> sol = solve(ODEProblem(apply_hamiltonian_sled!, m0, (0, TRF), (ω1, 1, ω0, R1s, T2s, G)), Tsit5())
retcode: Success
Interpolation: specialized 4th order "free" interpolation
t: 3-element Vector{Float64}:
 0.0
 7.475414666720001e-5
 0.0001
u: 3-element Vector{Vector{Float64}}:
 [1.0]
 [0.6313928231811967]
 [0.4895365449661915]

julia> using Plots

julia> plot(sol, labels=["zs"], xlabel="t (s)", ylabel="m(t)");



```
"""
function apply_hamiltonian_sled!(∂m∂t, m, p::Tuple{Real,Real,Real,Real,Real,Function}, t)
    ω1, B1, ω0, R1s, T2s, g = p

    xy = quadgk(τ -> g((t - τ) / T2s), 0, t, order=7)[1]
    ∂m∂t[1] = -B1^2 * ω1^2 * xy * m[1] + R1s * (1 - m[1])
    return ∂m∂t
end

function apply_hamiltonian_sled!(∂m∂t, m, p::Tuple{Function,Real,Any,Real,Real,Function}, t)
    ω1, B1, ω0, R1s, T2s, g = p

    xy = quadgk(τ -> ω1(τ)^2 * g((t - τ) / T2s), 0, t, order=7)[1]
    ∂m∂t[1] = -B1^2 * xy * m[1] + R1s * (1 - m[1])
    return ∂m∂t
end

function apply_hamiltonian_sled!(∂m∂t, m, p::Tuple{Real,Real,Real,Real,Real,Real,Real,Real,Real,Function}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1 * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1  * m[1] - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1f * m[5]

    ∂zs∂t = - B1^2 * ω1^2 * quadgk(τ -> g((t - τ) / T2s), 0, t, order=7)[1]
    ∂m∂t[4] = ∂zs∂t * m[4] + Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s * R1s * m[5]
    return ∂m∂t
end

function apply_hamiltonian_sled!(∂m∂t, m, p::Tuple{Function,Real,Real,Real,Real,Real,Real,Real,Real,Function}, t)
    ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g = p

    ∂m∂t[1] = - R2f * m[1] - ω0  * m[2] + B1 * ω1(t) * m[3]
    ∂m∂t[2] =   ω0  * m[1] - R2f * m[2]
    ∂m∂t[3] = - B1 * ω1(t)  * m[1] - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1f * m[5]

    ∂zs∂t = -B1^2 * quadgk(τ -> ω1(τ)^2 * g((t - τ) / T2s), 0, t, order=7)[1]
    ∂m∂t[4] = ∂zs∂t * m[4] + Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s * R1s * m[5]
    return ∂m∂t
end

function apply_hamiltonian_sled!(∂m∂t, m, p::Tuple{Function,Real,Function,Real,Real,Real,Real,Real,Real,Function}, t)
    ω1, B1, φ, m0s, R1f, R2f, Rx, R1s, T2s, g = p

    ∂m∂t[1] = - R2f * m[1] + B1 * ω1(t) * cos(φ(t)) * m[3]
    ∂m∂t[2] = - R2f * m[2] - B1 * ω1(t) * sin(φ(t)) * m[3]
    ∂m∂t[3] = - B1 * ω1(t) * cos(φ(t)) * m[1] + B1 * ω1(t) * sin(φ(t)) * m[2] - (R1f + Rx * m0s) * m[3] + Rx * (1 - m0s) * m[4] + (1 - m0s) * R1f * m[5]

    ∂zs∂t = -B1^2 * quadgk(τ -> ω1(τ)^2 * g((t - τ) / T2s), 0, t, order=7)[1]
    ∂m∂t[4] = ∂zs∂t * m[4] + Rx * m0s  * m[3] - (R1s + Rx * (1 - m0s)) * m[4] + m0s * R1s * m[5]
    return ∂m∂t
end