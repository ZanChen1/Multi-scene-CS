addpath('../Enc');
addpath('../Dec');
addpath('../channel');
addpath(genpath('../BCS'));
addpath('../NLR')
addpath(genpath('../TVNLR'))
addpath(genpath('../../DLAMP_Toolbox/gampmatlab'));
addpath(genpath('../../DLAMP_Toolbox'));
warning off
addpath('../../matconvnet-1.0-beta25/matlab/mex');
addpath('../../matconvnet-1.0-beta25/matlab');
addpath('../../matconvnet-1.0-beta25/matlab/simplenn');
addpath('../NoiseLevelEstimate1');
addpath(genpath('../../Trained_Weights'));
run('../../matconvnet-1.0-beta25/matlab/vl_setupnn.m');
%%
%rng(0)
randn('state',1);
rand('state',1);
global global_time
global_time = 0;
%%
Test_set = 'Set12'; % Waterloo_crop,set8
Test_image_dir = ['../../Test_Images/Set/',Test_set,'/'];
%Test_image_dir = ['../../Test_Images/Set/Set12'];
foldname = dir(Test_image_dir);
foldname = foldname(3:end);


%% load original images
for kk = 1:length(foldname)
    dir_name1 = dir(fullfile(Test_image_dir,foldname(kk).name,'*.png'));
    dir_name2 = dir(fullfile(Test_image_dir,foldname(kk).name,'*.tif'));
    dir_name = [dir_name1; dir_name2];
    
    %% denoize_choice
    for j=[3]
        denoize_choice = j;
        %% denoize_choice
        % 1: DnCNN_20Layers_10cases
        % 2: DnCNN_17Layers_10cases
        % 3: DnCNN_20Layers_3x3_17cases
        % 4: RCAN_5x5_dilated_17cases
        % 5: RCAN_5x5_dilated_10cases
        % 6: RCAN_3x3_dilated_17cases
        % 7: RCAN_5x5_17cases
        % 8: EDSR_5x5_17cases
        % 9: EDSR_3x3_10cases
        % 10: BM3D
        % 11: NLR-CS
        % 12: TVNLR
        % 13: BCS-SPL 
        % 14: RCAN_5x5_dilated_1case
        % 15: RCAN_5x5_dilated_17cases+
        % 16: RCAN_7x7_dilated_17cases
        %%
        
        for i= 1:6
            measure.rate = 0.05*i;
           
            for k = 1:length(dir_name)
                measure.Test_image_dir = fullfile(Test_image_dir, foldname(kk).name, dir_name(k).name);
                measure.Image_name=dir_name(k).name;
                ori_im = double(imread(measure.Test_image_dir));
                
                %%
                [measure,~] = para_set(measure, denoize_choice);
                rec_im = Enc_main(ori_im, measure);
                
                %%
                fprintf('Denoiser:%s, Rate:%f, Image_name:%s, PSNR:%f, SSIM:%f \n', measure.denoize_name, measure.rate, measure.Image_name,  csnr(rec_im, ori_im, 8,0, 0), cal_ssim(rec_im, ori_im, 0, 0));
                fp = fopen(['../results/', Test_set, '/', measure.denoize_name,'/', foldname(kk).name,'.csv'],'a');
                fprintf(fp,'%s, %f, %s, %f, %f \n', measure.denoize_name, measure.rate, measure.Image_name,  csnr(rec_im, ori_im, 8,0, 0), cal_ssim(rec_im, ori_im, 0, 0));
                fclose(fp);
                fp = fopen(['../results/', Test_set, '/', measure.denoize_name,'/', foldname(kk).name,'.txt'],'a');
                fprintf(fp,'Denoiser, %s, Rate, %f, Image_name, %s, PSNR, %f, SSIM, %f \n', measure.denoize_name, measure.rate, measure.Image_name,  csnr(rec_im, ori_im, 8,0, 0), cal_ssim(rec_im, ori_im, 0, 0));
                fclose(fp);
                %pip_imshow(ori_im,50, 70)
                %pip_imshow(rec_im,80, 120)
                
                %%
            end
        
        end
        
    end
    
end
fprintf('Time:%f\n',global_time/8);
