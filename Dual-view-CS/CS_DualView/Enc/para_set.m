function [measure,quantize] = para_set(measure, denoize_choice)

measure.ori_im_1 = double(imread(measure.Test_image_dir_1));
measure.ori_im_2 = double(imread(measure.Test_image_dir_2));
%measure.ori_im_2 = zeros(size(measure.ori_im_2));
quantize = [];
%% measurement parameter setting
if strcmp(measure.Test_set_name, 'BSD68')
    measure.ori_im_1 = measure.ori_im_1(1:end-1, 1:end-1);
    measure.ori_im_2 = measure.ori_im_2(1:end-1, 1:end-1);
    
    measure.block_height = 32; %% 
    measure.block_width = 32;   %% 
    
else
    measure.block_width = 64;
    measure.block_height = 64;
end
[image_height, image_width]=size(measure.ori_im_1);
measure.image_height = image_height;
measure.image_width = image_width; %divide the whole image into small blocks
measure.rate_allocation = ceil(measure.image_width*measure.image_height*measure.rate);

q=1:(measure.image_width*measure.image_height);
step(1,1) = 1;
step(1,2) = measure.rate_allocation;
measure.OMEGA = q(step(1,1):step(1,2));

measure.P_image=randperm(measure.image_height*measure.image_width);
measure.P_block=randperm(measure.block_width*measure.block_height);
measure.length = measure.image_width*measure.image_height;


%%
denoise_name_all = {'DnCNN_20Layers_10cases', 'DnCNN_17Layers_10cases',...
    'DnCNN_20Layers_3x3_17cases',...
    'RCAN_5x5_dilated_17cases', 'RCAN_5x5_dilated_10cases',...
    'RCAN_3x3_dilated_17cases', 'RCAN_5x5_17cases',...
    'EDSR_5x5_17cases', 'EDSR_3x3_10cases',...
    'BM3D', 'NLR-CS', 'TVNLR', 'BCS-SPL', 'RCAN_5x5_dilated_1case','RCAN_5x5_dilated_17cases+', ...
    'RCAN_7x7_dilated_17cases','ADMM-Net','MWCNN17','MWCNN24','NCRCAN','RCAN1B','DPIR','MWDNN','Restormer'};

measure.denoize_name = denoise_name_all{denoize_choice};
measure.model = 'Bernoulli';   %% Hadarmad or Bernoulli or Diffraction or Cartesian or Gaussian


end
