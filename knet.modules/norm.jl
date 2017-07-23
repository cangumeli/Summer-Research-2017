# All default normalization layers
# will be placed here
abstract AbstractBatchNorm <: KnetModule

# requires params and moments to be set
function forward(context, bn::AbstractBatchNorm, x)
   mode = bn.mode
   assert(mode in [:train, :test])
   #=assert(:gamma in fieldnames(bn))
   assert(:beta in fieldnames(bn))
   assert(:running_mean in fieldnames(bn))
   assert(:running_var in fieldnames(bn))
   assert(:affine in fieldnames(bn))=#
   tx = typeof(AutoGrad.getval(x))
   if typeof(bn.running_mean) !== tx
      bn.running_mean = tx(bn.running_mean)
   end
   if typeof(bn.running_var) !== tx
      bn.running_var = tx(bn.running_var)
   end
   red_dims = let nd = ndims(x)
      if nd == 4
         (1,2,4)
      elseif nd == 2
         (2,)
      else
         error("AbstractBatchNorm only supports 2 and 4 dimensional inputs")
      end
   end
   x_hat = nothing
   if mode === :test
      x_hat = (x .- bn.running_mean) ./ sqrt(bn.running_var .+ bn.eps)
   else
      # Do the computation
      m = 1
      for i in red_dims
         m *= size(x, i)
      end
      mu = sum(x, red_dims) ./ m
      x_mu = x .- mu
      sigma2 = sumabs2(x_mu, red_dims) ./ m
      x_hat = x_mu ./ sqrt(sigma2 .+ bn.eps)
      # Update the running stats
      bn.running_mean = bn.momentum * bn.running_mean + (1 - bn.momentum) * AutoGrad.getval(mu)
      bn.running_var = bn.momentum * bn.running_var + (1 - bn.momentum) * AutoGrad.getval(sigma2)
   end

   if bn.affine
      @po bn.gamma .* x_hat .+ bn.beta
   else
      x_hat
   end
end

type BatchNorm2 <: AbstractBatchNorm
   gamma
   beta
   running_mean
   running_var
   affine::Bool
   mode::Symbol
   eps
   momentum
   BatchNorm2(output::Int; affine=true, dtype=Array{Float32}, mode=:train, eps=1e-9, momentum=.9) =
      let dims = (output, 1)
         new(
            affine ? Parameter(dtype(ones(dims...))) : nothing,
            affine ? Parameter(dtype(zeros(dims...))) : nothing,
            dtype(zeros(dims...)),
            dtype(ones(dims...)),
            affine, mode, eps, momentum
         )
      end
end

type BatchNorm4 <: AbstractBatchNorm
   gamma
   beta
   running_mean
   running_var
   affine::Bool
   mode::Symbol
   eps
   momentum
   BatchNorm4(output::Int; affine=true, dtype=Array{Float32}, mode=:train, eps=1e-9, momentum=.9) =
      let dims = (1, 1, output, 1)
         new(
            affine ? Parameter(dtype(ones(dims...))) : nothing,
            affine ? Parameter(dtype(zeros(dims...))) : nothing,
            dtype(zeros(dims...)),
            dtype(ones(dims...)),
            affine, mode, eps, momentum
         )
      end
end
