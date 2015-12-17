function ancestors(node::Array{Any,1},nd::Int)
    anc=Array(Int,0)
    parents=node[nd].parents
    append!(anc,parents)
    done=false
    while !done
        newparents=Array(Int,0)
        for i in parents
            append!(newparents,node[i].parents)
        end
        if isempty(newparents)
            done=true
        else
            append!(anc,newparents)
            parents=newparents
        end
    end
    return sort(union(anc);rev=true)

end
export ancestors


function Parameters(net)
#    intersect(ancestors(net.node,net.FunctionNode),find(map(x->x.returnderivative,net.node))) # node indices that are parameters
    intersect(ancestors(net.node,net.FunctionNode),find(map(x->x.returnderivative,net.node[net.validnodes]))) # node indices that are parameters
end
export Parameters


function ForwardPassList!(net;ExcludeNodes=[])
    forwardlist=Array(Int,0)
    node=net.node
    N=length(node)
    exclude=Array(Int,length(ExcludeNodes))
    for i=1:length(ExcludeNodes)
        if isa(ExcludeNodes[i],ADnode)
            exclude[i]=ExcludeNodes[i].index
        else
            exclude[i]=ExcludeNodes[i]
        end
    end

    for i in intersect(setdiff(1:N,exclude),net.validnodes)
        thisnode=node[i]
        if length(findin(thisnode.parents,exclude))>0 # can be precomputed
            push!(exclude,i)# check
        else
            if !(thisnode.input)
                push!(forwardlist,i)
            end
        end
    end
    net.ForwardPassList=forwardlist
#    return net
end
export ForwardPassList!





