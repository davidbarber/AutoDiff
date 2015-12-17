function DemoSquare()
    #differentiate:   f(x)=sum(exp(-x.^2))
    # (c) David Barber, University College London 2015

    #using CUDArt


    StartCode()
    x=ADvariable()
    y=ADsquare(x)
    loss=ADsum(y)

    # instantiate parameter nodes and inputs:
    value=Array(Any,NodeCounter()) # function values on the nodes

D=20
N=20
    xx=ones(Float64,D,N); 
    value[x.index]=CUDArt.CudaArray(xx)

    (value,auxvalue,gradient,messvalue,net)=compile(value,Node();debug=false); # compile the DAG and preallocate memory

    ADeval!(value,net; auxvalue=auxvalue,gradient=gradient,messvalue=messvalue,debug=false,doReverse=true)
    println(to_host(value[x.index]))   
    println(to_host(value[y.index]))
    println(to_host(value[loss.index]))

#for i=1:1
#@time ADeval!(value,net; auxvalue=auxvalue,gradient=gradient,messvalue=messvalue,debug=false,doReverse=true)
#end
#println(to_host(value[loss.index]))


tic()
@time println(sum((xx.^2))) # function
#sum(AA,1) # gradient
toc()

    #net.FunctionNode=sm.index
    #gradcheck(value,net;showgrad=true) # use a small number of datapoints and small network to check the gradient, otherwise this will be very slow

   end
