macro gpu(_ex); if gpu()>=0; esc(_ex); end; end

macro cuda(lib,fun,x...)        # give an error if library missing, or if error code!=0
    if Libdl.find_library(["lib$lib"], []) != ""
        if VERSION >= v"0.6-"
            fx = Expr(:call, :ccall, ("$fun","lib$lib"), :UInt32, x...)
        else
            fx = Expr(:ccall, ("$fun","lib$lib"), :UInt32, x...)
        end
        msg = "$lib.$fun error "
        err = gensym()
        # esc(:(if ($err=$fx) != 0; warn($msg, $err); Base.show_backtrace(STDOUT, backtrace()); end))
        esc(:(if ($err=$fx) != 0; error($msg, $err); end; Knet.@gs))
    else
        Expr(:call,:error,"Cannot find lib$lib, please install it and rerun Pkg.build(\"Knet\").")
    end
end

macro cuda1(lib,fun,x...)       # return -1 if library missing, error code if run
    if Libdl.find_library(["lib$lib"], []) != ""
        if VERSION >= v"0.6-"
            fx = Expr(:call, :ccall, ("$fun","lib$lib"), :UInt32, x...)
        else
            fx = Expr(:ccall, ("$fun","lib$lib"), :UInt32, x...)
        end
        err = gensym()
        esc(:($err=$fx; Knet.@gs; $err))
    else
        -1
    end
end

macro knet8(fun,x...)       # error if libknet8 missing, nothing if run
    if libknet8 != ""
        if VERSION >= v"0.6-"
            fx = Expr(:call, :ccall, ("$fun",libknet8), :Void, x...)
        else
            fx = Expr(:ccall, ("$fun",libknet8), :Void, x...)
        end
        err = gensym()
        esc(:($err=$fx; Knet.@gs; $err))
    else
        Expr(:call,:error,"Cannot find libknet8, please rerun Pkg.build(\"Knet\").")
    end
end

const Cptr = Ptr{Void}

# Create a thread-safe GPU library
# which is more stateless to avoid race conditions
let CUBLAS=nothing, CUDNN=nothing
    global gpu, gpuCount, cublashandle, cudnnhandle, cudaRuntimeVersion, cudaDriverVersion
    
    gpu() = cudaGetDevice()

    function gpuCount()
        try
	    p=Cuint[0]
            # @cuda does not stay quiet so we use @cuda1 here
            # This code is only run once if successful, so nvmlInit here is ok
            @cuda1("nvidia-ml",nvmlInit,())
            @cuda1("nvidia-ml",nvmlDeviceGetCount,(Ptr{Cuint},),p)
            # Let us keep nvml initialized for future ops such as meminfo
            # @cuda1("nvidia-ml",nvmlShutdown,())
	    Int(p[1])
        catch
	    0
        end
    end

    function gpu(i::Int)
        if 0 <= i < gpuCount()
           cudaSetDevice(i)
           cudaRuntimeVersion = (p=Cint[0];@cuda(cudart,cudaRuntimeGetVersion,(Ptr{Cint},),p);Int(p[1]))
            cudaDriverVersion  = (p=Cint[0];@cuda(cudart,cudaDriverGetVersion, (Ptr{Cint},),p);Int(p[1]))
            i
        else
            -1
        end
    end

    function gpu(usegpu::Bool)
        if usegpu && gpuCount() > 0
            pick = free = same = -1
            for i=0:gpuCount()-1
                mem = nvmlDeviceGetMemoryInfo(i)
                if mem[2] > free
                    pick = i
                    free = mem[2]
                    same = 1
                elseif mem[2] == free
                    # pick one of equal devices randomly
                    rand(1:(same+=1)) == 1 && (pick = i)
                end
            end
            gpu(pick)
        else
            for i=0:gpuCount()-1
                @cuda(cudart,cudaDeviceReset,())
            end
            gpu(-1)
        end
    end

    function cublashandle(dev=gpu())
        if dev==-1; error("No cublashandle for CPU"); end
        i = dev+2
        if CUBLAS == nothing; CUBLAS=Array{Any}(gpuCount()+1); end
        if !isassigned(CUBLAS,i); CUBLAS[i]=cublasCreate(); end
        return CUBLAS[i]
    end

    function cudnnhandle(dev=gpu())
        if dev==-1; error("No cudnnhandle for CPU"); end
        i = dev+2
        if CUDNN == nothing; CUDNN=Array{Any}(gpuCount()+1); end
        if !isassigned(CUDNN,i); CUDNN[i]=cudnnCreate(); end
        return CUDNN[i]
    end
    
end


cudaGetDevice()=(d=Cint[-1];@cuda(cudart,cudaGetDevice,(Ptr{Cint},),d);d[1])
cudaSetDevice(i::Int) = @cuda(cudart,cudaSetDevice, (Cint,), i)
cudaGetMemInfo()=(f=Csize_t[0];m=Csize_t[0]; @cuda(cudart,cudaMemGetInfo,(Ptr{Csize_t},Ptr{Csize_t}),f,m); (Int(f[1]),Int(m[1])))
cudaDeviceSynchronize()=@cuda(cudart,cudaDeviceSynchronize,())

function nvmlDeviceGetMemoryInfo(i=gpu())
    0 <= i < gpuCount() || return nothing
    dev = Cptr[0]
    mem = Array{Culonglong}(3)
    @cuda("nvidia-ml","nvmlDeviceGetHandleByIndex",(Cuint,Ptr{Cptr}),i,dev)
    @cuda("nvidia-ml","nvmlDeviceGetMemoryInfo",(Cptr,Ptr{Culonglong}),dev[1],mem)
    ntuple(i->Int(mem[i]),length(mem))
end

function cublasCreate()
    handleP = Cptr[0]
    @cuda(cublas,cublasCreate_v2, (Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cublas,cublasDestroy_v2, (Cptr,), handle))
    global cublasVersion = (p=Cint[0];@cuda(cublas,cublasGetVersion_v2,(Cptr,Ptr{Cint}),handle,p);Int(p[1]))
    return handle
end

function cudnnCreate()
    handleP = Cptr[0]
    @cuda(cudnn,cudnnCreate,(Ptr{Cptr},), handleP)
    handle = handleP[1]
    atexit(()->@cuda(cudnn,cudnnDestroy,(Cptr,), handle))
    global cudnnVersion = Int(ccall((:cudnnGetVersion,:libcudnn),Csize_t,()))
    return handle
end


