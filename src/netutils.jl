#function ancestors(node::Array{Any,1},nd::Int)
#    anc=Array(Int,0)
#    parents=node[nd].parents
#    append!(anc,parents)
#    done=false
#    while !done
#        newparents=Array(Int,0)
#        for i in parents
#            append!(newparents,node[i].parents)
#        end
#        if isempty(newparents)
#            done=true
#        else
#            append!(anc,newparents)
#            parents=newparents
#        end
#    end
#    return sort(union(anc);rev=true)
#
#end
#export ancestors



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
        newparents=setdiff(newparents,anc)
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
#    intersect(ancestors(net.node,net.FunctionNode),find(map(x->x.returnderivative,net.node[net.validnodes]))) # node indices that are parameters

    good=[]
    for i in net.validnodes
        if net.node[i].returnderivative
            push!(good,i)
        end
    end
    intersect(ancestors(net.node,net.FunctionNode),good) # node indices that are parameters
end


    export Parameters

    function HasParents(node::ADnode)
        if length(node.parents)>0
            return true
        else
            return false
        end
    end
    export HasParents


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
#            if !(thisnode.input)
            if HasParents(thisnode)
                push!(forwardlist,i)
            end
        end
    end
    net.ForwardPassList=forwardlist
#    return net
end
export ForwardPassList!





