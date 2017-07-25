# as very simple cnn model
function add_conv!(weights, h, w, i, o)
   w = randn(h, w, i, o) * sqrt(2.0 / (h * w * o))
   b = zeros(1, 1, o, 1)
   push!(weights, w, b)
end

function add_linear!(weights, i, o)
   w = randn(o, i) * sqrt(2.0 / (o * i))
   b = zeros(o, 1)
   push!(weights, w, b)
end


function init_model()
   params = []
   add_conv!(params, 3, 3, 3, 8)
   add_conv!(params, 3, 3, 8, 16)
   add_conv!(params, 3, 3, 16, 32)
   add_linear!(params, 8 * 8 * 32, 200)
   add_linear!(params, 200, 10)
   params
end

function predict(w, x)
   o = relu(conv4(w[1], x; stride=2, padding=1)) .+ w[2]
   o = relu(conv4(w[3], o; stride=2, padding=1)) .+ w[4]
   o = relu(conv4(w[5], o; padding=1)) .+ w[6]
   o = relu(w[7] * mat(p)) .+ w[8]
   w[end-1] * o .+ w[end]
end

loss(w, x, y) = -sum(y .* logp(predict(w, x))) ./ size(y, 2)

lossgrad = grad(loss)
params = init_model()

# The multi-gpu logic
list_devices() = 0:Knet.gpuCount()-1
dtype = KnetArray{Float32}
devices = list_devices()
assert(length(devices) == Threads.@nthreads)
# replicate the model
replicas = Array{Any, 1}([nothing for _ in devices])

Threads.@threads for (i, d) in enumerate(devices)
   Knet.gpu(d)
   replicas[i] = map(dtype, params)
end


#Threads.@threads for i = 1:
#=gpus = gpulist()
g = Dict{Any, Any}()
w = init_model()
Threads.@threads for (i, d) in enumerate(gpus)
   Knet.gpu(d)
   w_ = map(x->KnetArray{Float32}(x), w)
   g[d] = map(x->Array{Float32}(x), lossgrad(...))
   for i = 1:length(g[d])
      update!(w_, g[d][i], opt)
   end
end=#
