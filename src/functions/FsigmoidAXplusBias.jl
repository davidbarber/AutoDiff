# Standard sigmoid layer: f(A,x,b)=1/(1+exp(-(A*x+b)))
# Note that the bias (a column vector) gets expanded here to match the dimension of A*x

# To make the CPU better performing, I think one would need to call c code that does inplace computation of Hadamard products. Essentially this is what is happending in the GPU code.

#FsigmoidAXplusBias(A,X,b)=(1./(1+exp(-(A*X+b*ones(1,size(X,2))))),nothing)

#TODO: don't need aux to be a tuple since we only use aux[1] 
FsigmoidAXplusBias(A,X,b)=(1./(1+exp(-(A*X+b*ones(1,size(X,2))))),(zeros(size(A,1),size(X,2)),nothing)) # allocate memory for inplace gradients

FsigmoidAXplusBias_inplace(value,aux,A,X,b)=copy!(value,1./(1+exp(-(A*X+b*ones(1,size(X,2))))))

#TODO; Need to do similar inplace computations for the other transfer functions
function DsigmoidAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X,b)
    copy!(faux_c[1],f_c)
    A_elmult_B_update!(-1.0,f_c,f_c,1.0,faux_c[1])
    A_elmult_B_update!(1.0,grad_c,faux_c[1],0.0,faux_c[1])

    if derivativeIDX==1
        #axpy!(1.0,(grad_c.*f_c.*(1-f_c))*X',grad_n) # this bad since it creates creates temporary storage which will need to be collected
        BLAS.gemm!('N','T',1.0,faux_c[1],X,1.0,grad_n)
    elseif derivativeIDX==2
        #axpy!(1.0,A'*(grad_c.*f_c.*(1-f_c)),grad_n)
        BLAS.gemm!('T','N',1.0,A,faux_c[1],1.0,grad_n)
    elseif derivativeIDX==3
        axpy!(1.0,sum(grad_c.*f_c.*(1-f_c),2),grad_n)
    end
end


if PROC=="GPU"
    function FsigmoidAXplusBias(A::CudaArray,X::CudaArray,b::CudaArray)
        return(sigmoid(FAXplusBias(A,X,b)[1]),nothing)
    end

    function FsigmoidAXplusBias_inplace(value,aux,A::CudaArray,X::CudaArray,b::CudaArray)
        FAXplusBias_inplace(value,aux,A,X,b)
        sigmoid!(value,value)
    end


    function DsigmoidAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray,b::CudaArray)
        tmp=CudaArray(Float64,size(grad_c)); fill!(tmp,0.0)
        tx1mx!(grad_c,f_c,tmp)
        if derivativeIDX==1
            gemm!('N','T',1.0,tmp,X,1.0,grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',1.0,A,tmp,1.0,grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float64,(size(X,2),1)); fill!(ons,1.0)
            gemm!('N','N',1.0,tmp,ons,1.0,grad_n)
            free(ons)
        end
        free(tmp)
    end

end


Derivative[FsigmoidAXplusBias]=DsigmoidAXplusBias
Inplace[FsigmoidAXplusBias]=FsigmoidAXplusBias_inplace

sigmoidAXplusBias(A,X,b)=ADFunction(FsigmoidAXplusBias,A,X,b)
export sigmoidAXplusBias



