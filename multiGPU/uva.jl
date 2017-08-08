# TODO: test this stuff

function checkP2P(did1::Int, did2::Int)::Bool
    access = Cint[0]
    @cuda(cudart, cudaDeviceCanAccessPeer, (Ptr{Cint}, Cint, Cint), access, did1, did2)
    res1 = Bool(access[1])
    @cuda(cudart, cudaDeviceCanAccessPeer, (Ptr{Cint}, Cint, Cint), access, did2, did1)
    res2 = Bool(access[2])
    res1 & res2
end

function enableP2P(did1::Int, did2::Int)
    gpu_temp = gpu()
    gpu(did1)
    @cuda(cudart, cudaDeviceEnablePeerAccess, (Cint, Cint), did2, 0)
    gpu(did2)
    @cuda(cudart, cudaDeviceEnablePeerAccess, (Cint, Cint), did1, 0)
    gpu(gpu_temp)
end


function all_pairs(f::Function, gpuList)
    ngpus = length(gpuList)
    for i = 1:ngpus-1
        for j = (i+1):ngpus
            f(gpuList[i], gpuList[j])
        end
    end
end

#=Enables p2p access for uva
if gpuList is nothing, then enables p2p between all gpus
otherwise, enable peer access between 
=#
function enableP2P(gpuList::Union{Array{Int, 1}, Void}=nothing)::Bool  
    # check the access status
    check = false
    all_pairs(gpuList) do d1, d2
        check &= checkP2P(d1, d2)
    end
    if ~check; return false; end
    
   # enable access between pers
    all_pairs(gpuList) do d1, d2
        enableP2P(d1, d2)
    end
    return true
end

