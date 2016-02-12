# Standard rectlin layer: f(A,x,b)=max(A*x+b,gamma(Ax+b)) where gamma=0.25
# Note that the bias (a column vector) gets expanded here to match the dimension of A*x

gamma=0.25

FkinklinAXplusBias(A,X,b)=begin; a=A*X+b*ones(1,size(X,2)); return (max(a,gamma*a),gamma+(1-gamma)*(a.>0)); end

function FkinklinAXplusBias_inplace(value,aux,A,X,b)
    a=A*X+b*ones(1,size(X,2))
    copy!(value,max(a,gamma*a))
    copy!(aux,gamma+(1-gamma)*(a.>0))
end

function DkinklinAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X,b)
    if derivativeIDX==1
        axpy!(1.0,(grad_c.*faux_c)*X',grad_n)
    elseif derivativeIDX==2
        axpy!(1.0,A'*(grad_c.*faux_c),grad_n)
    elseif derivativeIDX==3
        axpy!(1.0,sum(grad_c.*faux_c,2),grad_n)
    end
end

if PROC=="GPU" 

    function FkinklinAXplusBias_inplace(value,aux,A::CudaArray,X::CudaArray,b::CudaArray)
        FAXplusBias_inplace(value,[],A,X,b)
        copy!(aux,value)               
        kinklin!(value,value)
    end
    
    function DkinklinAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray,b::CudaArray)
        tmp=CudaArray(Float64,size(grad_c)); fill!(tmp,0.0)
        A_emult_Bg0!(grad_c,faux_c,tmp)        
        scale!((1-gamma),tmp)
        axpy!(gamma,grad_c,tmp)

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


    function DkinklinAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray{Float32},X::CudaArray{Float32},b::CudaArray{Float32})
        tmp=CudaArray(Float32,size(grad_c)); fill!(tmp,0.0)
        A_emult_Bg0!(grad_c,faux_c,tmp)        
        scale!((1-gamma),tmp)
        axpy!(gamma,grad_c,tmp)

        if derivativeIDX==1
            gemm!('N','T',Float32(1.0),tmp,X,Float32(1.0),grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',Float32(1.0),A,tmp,Float32(1.0),grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float32,(size(X,2),1)); fill!(ons,1.0)
            gemm!('N','N',Float32(1.0),tmp,ons,Float32(1.0),grad_n)
            free(ons)
        end
        free(tmp)
    end


    
end



Derivative[FkinklinAXplusBias]=DkinklinAXplusBias
Inplace[FkinklinAXplusBias]=FkinklinAXplusBias_inplace

kinklinAXplusBias(A,X,b)=ADnode(FkinklinAXplusBias,[A X b])
export kinklinAXplusBias


