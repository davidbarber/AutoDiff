global ADDIR
global PROC

PROC="CPU"

ADDIR=pwd()
cd("src")
include("$ADDIR/src/setup.jl")
#usegpu(true)
useproc(PROC)
using AutoDiff
cd("../")


#cd("src")
#include("src/setup.jl")
#usegpu(false)
#using AutoDiff
#cd("../")
