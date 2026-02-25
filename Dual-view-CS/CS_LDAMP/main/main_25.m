%%
clear
close all
%%
% addpath('../Enc');
% addpath('../Dec');
% addpath('../channel');
% addpath(genpath('../BCS'));
% addpath('../NLR')
% addpath(genpath('../TVNLR'))
% addpath(genpath('../../DLAMP_Toolbox/gampmatlab'));
% addpath(genpath('../../DLAMP_Toolbox/Algorithms'));
% addpath(genpath('../../DLAMP_Toolbox/Utils'));
% addpath(genpath('../../DLAMP_Toolbox/Packages/BM3D'));
% addpath(genpath('../../Matlab_Tools/MRI_lab'));
% %rmpath('/home/cz/Workspace/DLAMP_Toolbox/DnCNN_TIP2017-master/utilities');
% %rmpath('/home/cz/Workspace/CS_LDAMP/TVNLR/Utilities');
% %rmpath('/home/cz/Workspace/DLAMP_Toolbox/Packages')
% addpath(genpath('../../DLAMP_Toolbox/gampmatlab'));
% addpath(genpath('../../DLAMP_Toolbox'));
% warning off
% 
% addpath('../../Matlab_Tools/matconvnet-1.0-beta25/matlab');
% addpath('../../Matlab_Tools/matconvnet-1.0-beta25/matlab/simplenn');
% addpath('../../Matlab_Tools/matconvnet-1.0-beta25/matlab/mex');
% addpath('../NoiseLevelEstimate1');
% addpath(genpath('../../Trained_Weights'));
% run('../../Matlab_Tools/matconvnet-1.0-beta25/matlab/vl_setupnn.m');
addpath('../col_im');
%% python-env
% addpath('E:\Matlab\MWCNN\');
% if count(py.sys.path,'E:\Matlab\MWCNN\') == 0
%     insert(py.sys.path,int32(0),'E:\Matlab\MWCNN\');
% end
% py.importlib.reload(py.importlib.import_module('denoiser'));

%%
% addpath('E:\Matlab\MWDNN\');
% if count(py.sys.path,'E:\Matlab\MWDNN\') == 0
%     insert(py.sys.path,int32(0),'E:\Matlab\MWDNN\');
% end
% py.importlib.reload(py.importlib.import_module('denoiser_MWDNN'));
%
% addpath('E:\Matlab\DPIR\');
% if count(py.sys.path,'E:\Matlab\DPIR\') == 0
%     insert(py.sys.path,int32(0),'E:\Matlab\DPIR\');
% end
% py.importlib.reload(py.importlib.import_module('denoiser_DPIR'));

% addpath('E:\Matlab\sigma_estimate\');
% if count(py.sys.path,'E:\Matlab\sigma_estimate\') == 0
%     insert(py.sys.path,int32(0),'E:\Matlab\sigma_estimate\');
% end
% py.importlib.reload(py.importlib.import_module('Sigma_hat'));


% addpath('E:\Matlab\Trained_Weights\RCAN1B\');
% if count(py.sys.path,'E:\Matlab\Trained_Weights\RCAN1B\') == 0
%     insert(py.sys.path,int32(0),'E:\Matlab\Trained_Weights\RCAN1B\');
% end
% py.importlib.reload(py.importlib.import_module('rcan1b'));
%
% addpath('E:\Matlab\Trained_Weights\NCRCAN\');
% if count(py.sys.path,'E:\Matlab\Trained_Weights\NCRCAN\') == 0
%     insert(py.sys.path,int32(0),'E:\Matlab\Trained_Weights\NCRCAN\');
% end
% py.importlib.reload(py.importlib.import_module('ncrcan'));
%%
add_path();
global global_time
global_time = 0;
%%
Test_set = {'Set/Set11','urban100','Waterloo_crop/animal','Waterloo_crop/cityscape','Waterloo_crop/human'...
    'Waterloo_crop/landscape','Waterloo_crop/plant','Waterloo_crop/still-life','Waterloo_crop/transportation','128','256','512','1024'};
for j = [19]
    for sigma_test = 1
        for kk = [1] %[13,1]
            Test_image_dir = ['../../Test_Images/',Test_set{kk},'/'];
            foldname = dir(Test_image_dir);
            foldname = foldname(3:end);
            %% load original images
            for i = [25]  %[10,25,30,40,50] %[1,5,10,25,30,40,50]
                PNSR_sum = 0;
                SSIM_sum = 0;
                for seed = 1 %1:40
                    for k = 1:length(foldname)%1:length(foldname)
                        %                 if k==7
                        %                     seed = 1;
                        %                 end
                        rand('state',seed);
                        randn('state',seed);
                        measure.Test_image_dir = fullfile(Test_image_dir, foldname(k).name);
                        measure.Image_name=foldname(k).name;
                        ori_im = double(imread(measure.Test_image_dir));
                        measure.ori_im = ori_im;
                        %noise_im = gaussian_noise(ori_im, 0, sigma);
                        %csnr(noise_im, ori_im, 8,0, 0)  
                        %% denoize_choice
                        denoize_choice = j;
                     %% denoize_choice%%
                        % 1: DnCNN_20Layers_10cases      % 7: RCAN_5x5_17cases          % 13: BCS-SPL
                        % 2: DnCNN_17Layers_10cases       % 8: EDSR_5x5_17cases         % 14: RCAN_5x5_dilated_1case
                        % 3: DnCNN_20Layers_3x3_17cases    % 9: EDSR_3x3_10cases        % 15: RCAN_5x5_dilated_17cases+
                        % 4: RCAN_5x5_dilated_17cases       % 10: BM3D                  % 16: RCAN_7x7_dilated_17cases
                        % 5: RCAN_5x5_dilated_10cases        % 11: NLR-CS               % 17: ADMM-Net
                        % 6: RCAN_3x3_dilated_17cases         % 12: TVNLR,               %18 MWCNN17
                        %19 MWCNN24   %20:NCRCAN   %21:RCAN1B  %22:DPIR  
                        %23: MWDNN  %24:Restormer
                        measure.sigma_test = sigma_test;
                     %%
                        measure.rate = 0.01*i;
                        measure.Test_set_name = Test_set{kk};
                     %%
                        [measure,~] = para_set(measure, denoize_choice);
                        rec_im = Enc_main(measure.ori_im, measure);
                     %%
                        PNSR_sum = PNSR_sum + csnr(rec_im, measure.ori_im, 8,0, 0);
                        SSIM_sum = SSIM_sum + cal_ssim(rec_im, measure.ori_im, 0, 0);
                        %imwrite(uint8(rec_im),'House_org.png');
                        fprintf('Denoiser:%s, Rate:%f, measure_model:%s,Image_name:%s, PSNR:%f, SSIM:%f \n', measure.denoize_name, measure.rate, measure.model,measure.Image_name,  csnr(rec_im, measure.ori_im, 8,0, 0), cal_ssim(rec_im, ori_im, 0, 0));
                        fp = fopen(['../results/', Test_set{kk}, '/', measure.denoize_name,'.csv'],'a');
                        fprintf(fp,'%s, %f,%s, %s, %f, %f \n', measure.denoize_name, measure.rate, measure.model,measure.Image_name,  csnr(rec_im, measure.ori_im, 8,0, 0), cal_ssim(rec_im, measure.ori_im, 0, 0));
                        fclose(fp);
                        %figure(1),pip_imshow(ori_im,50, 70)  %boat 50 70 house 120,100 lena 130 150 Monarch 80 90

                    end
                    fp = fopen(['../results/', Test_set{kk}, '/', measure.denoize_name,'avg.csv'],'a');
                    fprintf(fp,'%s, %s, %f, %f \n',measure.rate,seed,PNSR_sum/length(foldname),SSIM_sum/length(foldname));
                    fclose(fp);
                end
            end
        end
    end
end

