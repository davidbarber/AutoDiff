global ADDIR
global PROC

PROC="GPU"

ADDIR=pwd()
cd("src")
include("$ADDIR/src/setup.jl")
#usegpu(true)
useproc("GPU")
using AutoDiff
cd("../")
