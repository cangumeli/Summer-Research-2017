using Images, FileIO

load_image = Images.load

function load_images(paths::Array{String,1})
   images = Array{Any, 1}([nothing for _ in paths])
   for (i, p) in enumerate(paths)
      images[i] = load(p)
   end
   return images
end

function to_raw_image(img; imgtype=RGB)
   colorview(imgtype, premute_dims(img, (3, 1, 2)))
end

function to_julia_array(img; dtype=Array{Float32})
   data = dtype(rawview(channelview(img)))
   permutedims(data, (2, 3, 1))
end

# Normalizes a regular julia array
function as_normalized(imgs, means)
   means = typeof(imgs)(means)
   if length(means) == size(imgs)[3] #channel
      imgs .- reshape(means, (1, 1, length(means), 1))
   else # batch
      imgs .- reshape(means, (1, 1, 1, length(means)))
   end
end

# Works with raw images
function rescale(img, new_size::Int)
   h, w = size(img)[end-1:end]
   if (w <= h && w == new_size) || (h <= w && h == size)
      return img
   end
   if w < h
      ow = new_size
      oh = div(new_size * h, w)
   else
      oh = new_size
      ow = div(new_size * w, h)
   end
   imresize(img, (oh, ow))
end

function hard_rescale(img, new_dims)
   imresize(img, new_dims)
end

function random_crop(img, crop_size)

end

function center_crop(img, crop_size)

end

function random_horizontal_flip!(img, p=0.5)

end

function pad_image(img; filler=0, pad_sper_dim=2, pad_dims=(1,2))

end
