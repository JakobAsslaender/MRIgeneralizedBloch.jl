using MRIgeneralizedBloch
using Documenter

DocMeta.setdocmeta!(MRIgeneralizedBloch, :DocTestSetup, :(using MRIgeneralizedBloch); recursive=true)

makedocs(;
    modules=[MRIgeneralizedBloch],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo="https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/blob/{commit}{path}#{line}",
    sitename="MRIgeneralizedBloch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JakobAsslaender/MRIgeneralizedBloch.jl",
)
