function load_Data(filename)
    filename = expanduser(filename)

    fid = open(filename);
    sIsProspa = Char.(read(fid, 8))
    version = Char.(read(fid, 4))
    if read(fid, Int32) != 501; error; end # 500 real; 501 complex; 502 double real; 503 xy_real; 504 xy_complex

    size1 = read(fid, Int32)
    size2 = read(fid, Int32)
    if read(fid, Int32) != 1; error; end
    if read(fid, Int32) != 1; error; end

    data = zeros(ComplexF32, size1, size2)
    for i = 1:length(data)
        data[i] = read(fid, Float32) + 1im * read(fid, Float32)
    end
    data = data[7:end - 7,:]
end

function load_spectral_integral(filename)
    data = load_Data(filename)
    data = fftshift(fft(data, 1), 1)

    M = vec(sum(+, data, dims=1))

    phase = angle(M[end])
    M = M .* exp(-1im * phase)
    M = real.(M)
end

function print_result(var_str, mean, std, unit_str)
    std_str = @sprintf("%1.1e", std)
    nDigits = 1 + floor(Int, log10(mean)) - floor(Int, log10(std))
    nDigits = max(nDigits, 1)
    fmt = string("%1.", nDigits, "e")
    mean_str = sprintf1(fmt, mean)
    println(string(var_str, " = ", mean_str, " \\pm ", std_str, " ", unit_str))
    return nothing
end

function gBloch_Hamiltonian_superLorentzian!(du, u, h, p, t)
    ωy, m0s, R1f, R1s, R2f, T2s, Rx, g = p
    du[1] = - R2f * u[1] + ωy * u[2]
    du[2] = - ωy  * u[1] - (R1f + Rx * m0s) * u[2] + Rx * (1 - m0s) * u[3] + (1 - m0s) * R1f * u[4]
    du[3] = - ωy^2 * quadgk(x -> g((t - x) / T2s) * h(p, x; idxs=3), eps(), t)[1] + Rx * m0s  * u[2] - (R1s + Rx * (1 - m0s)) * u[3] + m0s * R1s * u[4]
end

function Sled_Hamiltonian!(du, u, p, t)
    ωy, m0s, R1f, R1s, R2f, T2s, Rx, g = p
    du[1] = - R2f * u[1] + ωy * u[2]
    du[2] = - ωy  * u[1] - (R1f + Rx * m0s) * u[2] + Rx * (1 - m0s) * u[3] + (1 - m0s) * R1f * u[4]
    du[3] = - ωy^2 * u[3] * quadgk(x -> g((t - x) / T2s), 0, t)[1] + Rx * m0s  * u[2] - (R1s + Rx * (1 - m0s)) * u[3] + m0s * R1s * u[4]
end

function Graham_Hamiltonian_superLorentzian!(du, u, p, t)
    ωy, m0s, R1f, R1s, R2f, T2s, Rx, TRF = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0)[1]
    Rrf = f_PSD(TRF / T2s) * ωy^2 * T2s

    Linear_Hamiltonian!(du, u, (ωy, m0s, R1f, R1s, R2f, Rx, Rrf), t)
end

function Linear_Hamiltonian!(du, u, p, t)
    ωy, m0s, R1f, R1s, R2f, Rx, Rrf = p
    
    # Free precession
    du[1] = - R2f * u[1]
    du[2] = - (R1f + Rx * m0s) * u[2] + Rx * (1 - m0s)  * u[3] + (1 - m0s) * R1f * u[3]
    du[3] =   Rx * m0s  * u[2] - (R1s + Rx * (1 - m0s)) * u[3] + m0s  * R1s * u[3]
    
    # RF-pulse
    du[1] += ωy * u[2]
    du[2] -= ωy * u[1]
    du[3] -= Rrf * u[3]
end

function gBloch_IR_model(x, p, g_SLa, ω1, TRF, TI, R2f, model)
    (m0, m0f_inv, m0s, R1, T2s, Rx) = p
    R1f = R1
    R1s = R1
    
    m0f = 1 - m0s
    h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(4)

    M = zeros(Float64, length(TI), length(TRF))
    for i = 1:length(TRF)
        u0 = [0, m0f, m0s, 1]

        if model == :gBloch
            u = solve(DDEProblem(gBloch_Hamiltonian_superLorentzian!, u0, h, (0.0, TRF[i]), (ω1[i], m0s, R1f, R1s, R2f, T2s, Rx, g_SLa)), MethodOfSteps(DP8()))[end]
        elseif model == :Graham
            u = solve(ODEProblem(Graham_Hamiltonian_superLorentzian!, u0, (0.0, TRF[i]), (ω1[i], m0s, R1f, R1s, R2f, T2s, Rx, TRF[i])), Tsit5())[end]
        elseif model == :Sled
            u = solve(ODEProblem(Sled_Hamiltonian!, u0, (0.0, TRF[i]), (ω1[i], m0s, R1f, R1s, R2f, T2s, Rx, g_SLa)), Tsit5())[end]
        else
            error()
        end

        H = [-R1f - m0s * Rx        m0f * Rx R1f * m0f;
                    m0s * Rx -R1s - m0f * Rx R1s * m0s;
                        0            0         0]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - TRF[i] / 2)) * [m0f_inv * u[2],u[3],1])[1]
        end
    end
    return vec(M)
end

