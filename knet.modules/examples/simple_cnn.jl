include("../knet.modules.jl")
include("data.jl")

function loaddata()
   dtr, dts = data.cifar10()
   (xtrn, ytrn) = dtr
   (xtst, ytst) = dts
   mnt = mean(xtrn, (1, 2, 4))
   xtrn .-= mnt
   xtst .-= mnt
   println("Data is loaded")
   return (xtrn, ytrn), (xtst, ytst)
end

function next_batch(x, y; dtype=Array{Float32}, bs=128)
   batch_indices = rand(1:size(x, 4), bs)
   x_, y_ =  x[:, :, :, batch_indices], y[:, batch_indices]
   return dtype(x_), dtype(y_)
end

model = Sequential(
   Conv4(3, 3, 3, 8; padding=1, bias=false),
   BatchNorm4(8),
   relu,
   pool,
   Conv4(3, 3, 8, 16; padding=1, bias=false),
   BatchNorm4(16),
   relu,
   pool,
   Linear(8 * 8 * 16, 200; bias=false),
   BatchNorm2(200),
   relu,
   Dropout(.9),
   Linear(200, 10)
)

dtype = Array{Float32}
if Knet.gpu() >= 0
   gpu!(model)
   dtype = KnetArray{Float32}
end

lr = .1
model_params = parameters(model)
println(length(model_params))
dtrn, dtst = loaddata()
loss(y, ygold) = -sum(ygold .* logp(y)) ./ size(ygold, 2)
lossgrad = grad(model, loss)
optims = [Momentum(;lr=lr) for p in model_params]

# println(sort(map(x->x.index, model_params)))
#=dtype = Array{Float64}
cpu!(model; dtype=Float64)=#

function debug_modes(model)
   println("Modes: ", (map(x->x.mode, filter(x->isa(x, AbstractBatchNorm) || isa(x, Dropout),
      layers(model)))))
end

for i = 1:10000
   if i % 10 == 0; println("Iter: ", i); end
   # Perform a training step
   x, y = next_batch(dtrn[1], dtrn[2]; dtype=dtype)
   g = lossgrad(x, y)
   for (p, o) in zip(model_params, optims)
      update!(get_value(p), get_grad(p, g), o)
   end
   # Accuracy measurement
   if i % 100 == 0 || i == 1
      testing!(model)
      debug_modes(model)
      ncorrect = 0
      for j = 1:200:size(dtst[1], 4)
         x_, y_ = dtst[1][:, :, :, j:j+199], dtst[2][:, j:j+199]
         x_ = dtype(x_)
         y_ = dtype(y_)
         ypred = @run(model, x_)
         ncorrect += sum(y_ .* (ypred .== maximum(ypred,1)))
      end
      println("accuracy:", ncorrect/size(dtst[1], 4), " i:", i)
      training!(model)
      debug_modes(model)
   end
end
