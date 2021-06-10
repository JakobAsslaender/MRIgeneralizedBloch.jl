using MT_generalizedBloch
using Test

@testset "Greens_Interpolation" begin
    include("Greens_Interpolation.jl")
end

@testset "gBloch_Hamiltonian!" begin
    for i = 1:10 # test different random initializations
        include("gBloch_Hamiltonian.jl")
    end
end

@testset "FreePrecession_Hamiltonian!" begin
    for i = 1:10 # test different random initializations
        include("FreePrecession_Hamiltonian.jl")
    end
end

@testset "Graham_Hamiltonian_superLorentzian!" begin
    for i = 1:10 # test different random initializations
        include("Graham_Hamiltonian_superLorentzian.jl")
    end
end

@testset "Linear_Approximation" begin
    include("Linear_Approximation.jl")
end

@testset "gBloch_Hamiltonian! gradients" begin
    include("gBloch_Hamiltonian_Gradients.jl")
end

@testset "FreePrecession_Hamiltonian! gradients" begin
    include("FreePrecession_Hamiltonian_Gradients.jl")
end

@testset "Linaer_Approx gradients" begin
    include("Linaer_Approx_Gradients.jl")
end

@testset "Graham_Hamiltonian_superLorentzian! gradients" begin
    include("Graham_Hamiltonian_Gradients.jl")
end

@testset "calculate_magnetization_n_signal" begin
    include("calculate_magnetization_n_signal.jl")
end