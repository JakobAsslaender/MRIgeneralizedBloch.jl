using Pkg
Pkg.activate("docs")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()

using MRIgeneralizedBloch
using Documenter
using Literate
using Plots

# HTML Plotting Functionality
struct HTMLPlot
    p # :: Plots.Plot
end
const ROOT_DIR = joinpath(@__DIR__, "build")
const PLOT_DIR = joinpath(ROOT_DIR, "plots")
function Base.show(io::IO, ::MIME"text/html", p::HTMLPlot)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, string(UInt32(floor(rand() * 1e9)), ".html"))
    Plots.savefig(p.p, path)
    if get(ENV, "CI", "false") == "true" # for prettyurl
        print(io, "<object type=\"text/html\" data=\"../../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    else
        print(io, "<object type=\"text/html\" data=\"../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    end
end

# Notebook hack to display inline math correctly
function notebook_filter(str)
    re = r"(?<!`)``(?!`)"  # Two backquotes not preceded by nor followed by another
    return replace(str, re => "\$")
end

# Literate
OUTPUT = joinpath(@__DIR__, "src/build_literate")

files = [
    "tutorial_singlepulse.jl",
    "tutorial_pulsetrain.jl",
    "NLLS.jl",
    "OCT.jl",
    "Greens_functions.jl",
    "Simulation_ContinuousWave.jl",
    "Simulation_Pulse.jl",
    "Analyze_NMR_IR_Data.jl",
    "Analyze_NMR_PreSat_Data.jl",
    "Linear_Approximation.jl",
]

for file in files
    file_path = joinpath(@__DIR__, "src/", file)
    Literate.markdown(file_path, OUTPUT)
    Literate.notebook(file_path, OUTPUT, preprocess=notebook_filter; execute=false)
    Literate.script(file_path, OUTPUT)
end

DocMeta.setdocmeta!(MRIgeneralizedBloch, :DocTestSetup, :(using MRIgeneralizedBloch); recursive=true)

makedocs(;
    doctest=true,
    doctestfilters = [r"\s*-?(\d+)\.(\d{4})\d*\s*"], # Ignore any digit after the 4th digit after a decimal, throughout the docs
    modules=[MRIgeneralizedBloch],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo = Documenter.Remotes.GitHub("JakobAsslaender", "MRIgeneralizedBloch.jl"),
    sitename="MRIgeneralizedBloch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "build_literate/tutorial_singlepulse.md",
        "build_literate/tutorial_pulsetrain.md",
        "build_literate/NLLS.md",
        "build_literate/OCT.md",
        "Generalized Bloch Paper" => Any[
            "build_literate/Greens_functions.md",
            "build_literate/Simulation_ContinuousWave.md",
            "build_literate/Simulation_Pulse.md",
            "build_literate/Analyze_NMR_IR_Data.md",
            "build_literate/Analyze_NMR_PreSat_Data.md",
            "build_literate/Linear_Approximation.md",
        ],
        "API" => "api.md",
    ],
)

# Set dark theme as default independent of the OS's settings
run(`sed -i'.old' 's/var darkPreference = false/var darkPreference = true/g' docs/build/assets/themeswap.js`)

deploydocs(;
    repo="github.com/JakobAsslaender/MRIgeneralizedBloch.jl",
    push_preview=true,
)
