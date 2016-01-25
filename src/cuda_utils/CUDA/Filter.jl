# filter pointer
typealias cudnnFilterDescriptor_t Ptr{Void} # hold the description of a filter dataset 
export cudnnFilterDescriptor_t


function cudnnCreateFilterDescriptor()
filterDesc = cudnnFilterDescriptor_t[0]
@cudnncheck(:cudnnCreateFilterDescriptor,(Ptr{cudnnFilterDescriptor_t},),filterDesc)
return filterDesc[1]
end

function cudnnSetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t,dataType::Int,k,c,h,w)
@cudnncheck(:cudnnSetFilter4dDescriptor,(cudnnFilterDescriptor_t,Cint,Cint,Cint,Cint,Cint),filterDesc,dataType,k,c,h,w)
end

function cudnnGetFilter4dDescriptor(filterDesc::cudnnFilterDescriptor_t)
dataType = Cint[0]
k = Cint[0]
c = Cint[0]
h = Cint[0]
w = Cint[0]
@cudnncheck(:cudnnGetFilter4dDescriptor,(cudnnFilterDescriptor_t,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,dataType,k,c,h,w)
return(filterDesc,(k[1],c[1],h[1],w[1]))
end

function cudnnSetFilterNdDescriptor{T<:AbstractFloat}(filterDesc::cudnnFilterDescriptor_t,dataType::T,nbDims::UInt,filterDimA::Array{UInt,1})
dtype = cudnnDataTypeCheck(dataType)
@cudnncheck(:cudnnSetFilterNdDescriptor,(cudnnFilterDescriptor_t,Cint,Cint,Ptr{Cint}),filterDesc,dtype,nbDims,filterDimA)
end

function cudnnGetFilterNdDescriptor(filterDesc::cudnnFilterDescriptor_t,nbDimsRequested::UInt,filterDimA::Array{UInt,1})
dataType = Cint[0]
nbDims = Cint[0]
@cudnncheck(:cudnnGetFilterNdDescriptor,(cudnnFilterDescriptor_t,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint}),filterDesc,nbDimsRequested,dataType,nbDims,filterDimA)
dtype = cudnnDataTypeConvert(dataType[1])
return (filterDesc,dtype,nbDims[1],filterDimA)
end

function cudnnDestroyFilterDescriptor(filterDesc::cudnnFilterDescriptor_t)
@cudnncheck(:cudnnDestroyFilterDescriptor,(cudnnFilterDescriptor_t,),filterDesc)
end
