using PyCall
@pyimport torchfile

# https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
#
function _set_conv!(myconv, tconv)
   set_value!(myconv.weight, permutedims(
      tconv[:weight], (3,4,2,1)))
end

function _set_bn!(mybn, tbn)
   #=println(keys(tbn))
   println(size(tbn[:weight]))=#
   r(x) = reshape(x, (1, 1, size(x)[1], 1))
   set_value!(mybn.gamma, r(tbn[:weight]))
   set_value!(mybn.beta, r(tbn[:bias]))
   copy!(mybn.running_mean, r(tbn[:running_mean]))
   copy!(mybn.running_var, r(tbn[:running_var]))
end

# Debugging stuff
global _num_conv = 0
global _num_bnorm = 0

function _set_block!(mybl, tbl)
   global _num_conv, _num_bnorm
   layers = tbl[:modules][1][:modules][1][:modules]
   shortcut = tbl[:modules][1][:modules][2][:modules]
   for i = 1:length(layers)
      if layers[i][:torch_typename]() == "cudnn.SpatialConvolution"
         _num_conv += 1
         _set_conv!(mybl.layers.layers[i], layers[i])
      elseif layers[i][:torch_typename]() == "nn.SpatialBatchNormalization"
         _num_bnorm += 1
         _set_bn!(mybl.layers.layers[i], layers[i])
      end
   end
   # Load the shortcut
   if typeof(shortcut) == Array{Any, 1} && length(shortcut) == 2
      # println("Loaded shorchut")
      _set_conv!(mybl.shortcut.op.layers[1], shortcut[1])
      _set_bn!(mybl.shortcut.op.layers[2], shortcut[2])
   # else
      # println("Identity observed")
   end
end

function _set_linear!(myl, tl)
   wt = tl[:weight]
   bt = tl[:bias]
   bt = reshape(bt, (size(bt)[1], 1))
   set_value!(myl.weight, wt)
   set_value!(myl.bias, bt)
end


# filename and ResNet
function load_torch_weights!(resnet, filename)
   o = torchfile.load(filename)
   modules = o[:modules]
   conv1, bn1 = modules[1], modules[2]
   #=println(conv1[:torch_type]())
   println(bn1[:torch_type]())=#
   myconv1, mybn1 = resnet.inp.layers[1:2]
   _set_conv!(myconv1, modules[1])
   _set_bn!(mybn1, modules[2])
   # Flatten the layer-based grouping
   tblocks = []
   for i = 5:8
      push!(tblocks, modules[i][:modules]...)
   end
   # Early layer feature maps
   if 2 <= resnet.config.conv_features <= 5
      tblocks = tblocks[1:length(resnet.blocks)]
   end
   assert(length(tblocks) == length(resnet.blocks))
   # Set the block weights
   for (tblock, block) in zip(tblocks, resnet.blocks)
      _set_block!(block, tblock)
   end
   if resnet.out !== nothing
      _set_linear!(resnet.out.layers[end], modules[end])
   end
end

# TODO: support preactivation
# Downloads
function download_torch_weights(depth::Int; dirname="torch_weights")
   urls = Dict([
      18  => "https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7",
      34  => "https://d2j0dndfm35trm.cloudfront.net/resnet-34.t7",
      # TODO: reopen resnet 50 when after-add bnorm is supported
      # 50  => "https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7",
      101 => "https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7",
      152 => "https://d2j0dndfm35trm.cloudfront.net/resnet-152.t7"
   ])
   if ~haskey(urls, depth)
      error("Dept ", depth, " is not supported yet.")
   end
   dir() = string(pwd(), "/", dirname)
   uri(depth) = string(dir(), "/resnet-", depth)
   if ~isdir(dir())
      mkdir(dir())
   end
   if isfile(uri(depth))
      # println(uri(depth), " already exists.")
      return uri(depth)
   end
   download(urls[depth], uri(depth))
   return uri(depth)
end
