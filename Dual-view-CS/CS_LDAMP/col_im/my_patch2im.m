function Im = my_patch2im(Patch, img_size, patch_size, patch_step)


d = size(Patch,3);
col = zeros(patch_size(1)*patch_size(2), d);
for i = 1:d
    temp = Patch(:,:,i);
    col(:,i) = temp(:);
end

T = ones(size(col));
Im = col2imstep(col,img_size,patch_size,patch_step);
T = col2imstep(T,img_size,patch_size,patch_step);
Im = Im./T;

end