function greens_lorentzian(τ)
    exp(-τ)
end
function greens_lorentzian(τ, R2)
    greens_lorentzian(R2 * τ)
end
function greens_lorentzian(t, τ, R2)
    greens_lorentzian(R2 * (t-τ))
end

function greens_gaussian(τ)
    exp(-τ^2 / 2)
end
function greens_gaussian(τ, R2)
    greens_gaussian(R2 * τ)
end
function greens_gaussian(t, τ, R2)
    greens_gaussian(R2 * (t-τ))
end

function greens_superlorentzian(τ)
    quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0, 1)[1]
end

function dG_o_dT2s_x_T2s_superlorentzian(τ)
    quadgk(ct -> exp(-τ^2 * (3 * ct^2 - 1)^2 / 8) * (τ^2 * (3 * ct^2 - 1)^2 / 4), 0.0, 1.0)[1]
end
function dG_o_dT2s_x_T2s_superlorentzian(τ, R2)
    dG_o_dT2s_x_T2s_superlorentzian(R2 * τ)
end
function dG_o_dT2s_x_T2s_superlorentzian(t, τ, R2)
    dG_o_dT2s_x_T2s_superlorentzian(R2 * (t-τ))
end

function interpolate_greens_function(f, τmin, τmax)
    Fun(f, τmin..τmax)
    # x = τmin : 0.001 : τmax
    # return CubicSplineInterpolation(x, f.(x))
end