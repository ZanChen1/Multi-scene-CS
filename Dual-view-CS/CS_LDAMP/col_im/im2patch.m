function Patch = im2patch(im, Patch_size, Patch_step)

h_index_list = [1:Patch_step(1):size(im,1)-Patch_size(1),size(im,1)-Patch_size(1)+1];
w_index_list = [1:Patch_step(2):size(im,2)-Patch_size(2),size(im,2)-Patch_size(2)+1];
Patch = [];
k = 1;
for i=1:length(h_index_list)
    for j=1:length(w_index_list)
        h_index = h_index_list(i);
        w_index = w_index_list(j);
        Patch(:,:,k) = im(h_index:h_index+Patch_size(1)-1,w_index:w_index+Patch_size(2)-1);
        k = k+1;
        
    end
    
end


end