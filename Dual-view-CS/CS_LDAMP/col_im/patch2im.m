function Im = patch2im(Patch, Img_size, Patch_size, Patch_step)

h_index_list = [1:Patch_step(1):Img_size(1)-Patch_size(1),Img_size(1)-Patch_size(1)+1];
w_index_list = [1:Patch_step(2):Img_size(2)-Patch_size(2),Img_size(2)-Patch_size(2)+1];

Im = zeros(Img_size);
Cnt = zeros(Img_size);
k = 1;
for i=1:length(h_index_list)
    for j=1:length(w_index_list)
        h_index = h_index_list(i);
        w_index = w_index_list(j);
        tmp = Im(h_index:h_index+Patch_size(1)-1,w_index:w_index+Patch_size(2)-1);
        Im(h_index:h_index+Patch_size(1)-1,w_index:w_index+Patch_size(2)-1)=tmp+Patch(:,:,k);
        tmp = Cnt(h_index:h_index+Patch_size(1)-1,w_index:w_index+Patch_size(2)-1);
        Cnt(h_index:h_index+Patch_size(1)-1,w_index:w_index+Patch_size(2)-1)=tmp+1;
        k = k+1;
    end
end
Im = Im./Cnt;

end