function Greens_Lorentzian(τ)
    exp(-τ)
end
function Greens_Lorentzian(τ, R2)
    Greens_Lorentzian(R2 * τ)
end
function Greens_Lorentzian(t, τ, R2)
    Greens_Lorentzian(R2 * (t-τ))
end

function Greens_Gaussian(τ)
    exp(-τ^2 / 2)
end
function Greens_Gaussian(τ, R2)
    Greens_Gaussian(R2 * τ)
end
function Greens_Gaussian(t, τ, R2)
    Greens_Gaussian(R2 * (t-τ))
end

function Greens_superLorentzian(τ)
    quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
end
function Greens_superLorentzian(τ, R2)
    Greens_superLorentzian(R2 * τ)
end
function Greens_superLorentzian(t, τ, R2)
    Greens_superLorentzian(R2 * (t-τ))
end

function dG_o_dT2s_x_T2s_superLorentzian(τ)
    quadgk(ct -> exp(-τ^2 * (3 * ct^2 - 1)^2 / 8) * (τ^2 * (3 * ct^2 - 1)^2 / 4), 0.0, 1.0)[1]
end
function dG_o_dT2s_x_T2s_superLorentzian(τ, R2)
    dG_o_dT2s_x_T2s_superLorentzian(R2 * τ)
end
function dG_o_dT2s_x_T2s_superLorentzian(t, τ, R2)
    dG_o_dT2s_x_T2s_superLorentzian(R2 * (t-τ))
end

function interpolate_Greens_Function(f, τmin, τmax)
    Fun(f, τmin..τmax)
end