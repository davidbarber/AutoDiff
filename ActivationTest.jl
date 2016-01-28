workspace()
include("GPUStart.jl")
StartCode()
t = Tensor((1,1,100,100))
c = CUActivation(t)
net = EndCode()
net.value[t] = rand(100,100)
net = compile(net,backend="GPU")
CUDArt.init([0])
ADforward!(net)
ADbackward!(net)