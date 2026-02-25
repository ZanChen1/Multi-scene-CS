



ori_im = imread( 'lena.tif');
ori_im = double(ori_im);
ori_im = ori_im(:,:,1);
step_size = [12,12];

% Patch = im2colstep(ori_im,[64 64],step_size);
% T = ones(size(ori_im));
% T = im2colstep(T,[64,64],step_size);
% Im = col2imstep(Patch,[256,256],[64,64],step_size);
% T = col2imstep(T,[256,256],[64,64],step_size);
% Im = Im./T;

Patch2 = im2patch(ori_im,[64 64],step_size);
Im = patch2im(Patch2, [256,256],[64,64],step_size);
