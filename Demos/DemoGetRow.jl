useproc("CPU")

StartCode()
x=ADvariable() # global memory
y=x[1,:]+x[2,:]
loss=sum(y)
net=EndCode() # defines the graph

# parameters:
net.value[x]=randn(3,2)
net=compile(net,debug=true) # compile and preallocate memory
@gpu CUDArt.init([0])
@gpu net=convert(net,"GPU")

gradcheck(net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise
