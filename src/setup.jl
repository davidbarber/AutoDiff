global PROC

push!(LOAD_PATH,pwd())
#push!(LOAD_PATH, joinpath(pwd(), "src"))

#push all subdirectories from src
map(d -> push!(LOAD_PATH, joinpath(pwd(), "./", d)),
    filter(d -> isdir(joinpath("./", d)), readdir("./")))

#push!(LOAD_PATH, joinpath(pwd(), "Demos"))

#include("usegpu.jl")
include("useproc.jl")
