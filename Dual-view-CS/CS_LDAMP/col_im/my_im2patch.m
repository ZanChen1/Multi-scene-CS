function Patch = my_im2patch(ori_im, size_h, step_h)

col = im2colstep(ori_im,[size_h],[step_h]);
d = size(col,2);
Patch = zeros(size_h(1), size_h(2), d);
for i = 1:d
   Patch(:,:,i) = reshape(col(:,i), size_h); 
end

end