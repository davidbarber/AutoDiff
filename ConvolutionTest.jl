workspace()
include("GPUStart.jl")
StartCode()
t = Tensor((1,1,28,28))
f = Filters(3)
c = Convolution(t,f,(20,1))
net = EndCode()
net.value[t] = rand(28,28)
net.value[f] = rand(3,3)
net = compile(net,backend="GPU")
CUDArt.init([0])
ADforward!(net)
ADbackward!(net)