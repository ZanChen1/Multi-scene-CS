%%
clear,close all;
% cudnn_lib = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp';
% cudnn_bin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin';
% cudnn_path = [cudnn_lib, ';',  cudnn_bin, ';'];
% if isempty(getenv('PATH_BACKUP'))
%     Env_PATH = getenv('PATH');
%     setenv('PATH_BACKUP', [cudnn_path, getenv('PATH')]);
% else
%
% end
% setenv('PATH', getenv('PATH_BACKUP'));
%%
add_path()
%%
randn('state',0);
rand('state',0);

SR = [4,10,20,25,30,40,50];
measure.iterative_way = 1;
%%
Test_set = {'Set/Set8','Set/Set11','Waterloo_crop/animal','Waterloo_crop/cityscape','Waterloo_crop/human'...
    'Waterloo_crop/landscape','Waterloo_crop/plant','Waterloo_crop/still_life','Waterloo_crop/transportation','Set/little5','BSD68', 'urban100', 'Set/Set12', 'Set/Set11'};

for denoize_choice_1 = [10]  
    for denoize_choice_2 = [22,23]  
        for set = 14 %[12,13]
            Test_image_dir = ['../../Test_Images/',Test_set{set},'/'];
            foldname = dir(Test_image_dir);
            foldname = foldname(3:end);
            img_num = length(foldname(:,1));
            measure.Test_set_name = Test_set{set};

            if strcmp(measure.Test_set_name, 'Set/Set11')   %用set11做消融实验，解决图片是单数的问题（强行匹配两张大图，且所有图测两遍）
                img_num = img_num*2;
            end

            for kk= 1:7
                PSNR_sum = 0;
                SSIM_sum = 0;
                MSE_sum = 0;


                for k = 1:int32(img_num/2)
                    measure.rate = 0.01*SR(kk);
                    %% load original images
                    switch measure.Test_set_name
                        case 'Set/Set12'    %大小为512和512一起重建，256和256一起重建
                            k_1 = 2*k - 1;
                            k_2 = 2*k;

                        case 'Set/Set11'    %大小为512和512一起重建，256和256一起重建
                            k_1 = k;
                            k_2 = k_1+1;
                            if k ==1 
                                k_1 = 1;
                                k_2 = 2;
                            elseif k==2
                                k_1 = 2;
                                k_2 = 1;
                            end
                            if k_2 > 11
                                k_2 = 3;
                            end

                        otherwise 
                            k_1 = k;
                            k_2 = (img_num/2)+k;    
                    end

                    measure.Image_name_1=foldname(k_1).name;
                    measure.Image_name_2=foldname(k_2).name;
                    
                    measure.sigma_test = 1; 
                    seed_1 = 1;
                    seed_2 = 2;
                    %%
                    Test_image_dir_1 = fullfile(Test_image_dir, measure.Image_name_1);
                    Test_image_dir_2 = fullfile(Test_image_dir, measure.Image_name_2);
                    [measure_1,~] = para_set_individual(measure, denoize_choice_1, Test_image_dir_1,Test_image_dir_2, seed_1);
                    [measure_2,~] = para_set_individual(measure_1, denoize_choice_2, Test_image_dir_2,Test_image_dir_1, seed_2);

                    %         measure.Test_image_dir_1 = fullfile(Test_image_dir, measure.Image_name_1);
                    %         measure.Test_image_dir_2 = fullfile(Test_image_dir, measure.Image_name_2);
                    %         [measure,quantize] = para_set(measure, denoize_choice);
                    %         measure_2 = measure;
                    %         measure_1 = measure;
                    %         measure_1.ori_im = measure.ori_im_1;
                    %         measure_2.ori_im = measure.ori_im_2;   %%%%para_set的设置，measure_1与measure_2的设置应该尽量的一样么？
                    %%
                
                    [rec_im_1,rec_im_2,MSE_im_1,MSE_im_2] = Enc_main(measure_1.ori_im, measure_2.ori_im, measure_1, measure_2);
                    
                    PSNR_sum = PSNR_sum+csnr(rec_im_1,  measure_1.ori_im, 8,0, 0)+csnr(rec_im_2,  measure_2.ori_im, 8,0, 0);
                    SSIM_sum = SSIM_sum+cal_ssim(rec_im_1, measure_1.ori_im, 0, 0)+cal_ssim(rec_im_2, measure_2.ori_im, 0, 0);
                    MSE_sum = MSE_sum+MSE_im_1+MSE_im_2;
                    fprintf('Denoiser_1:%s,Denoiser_2:%s, Rate:%f, measure_model:%s \n',measure_1.denoize_name,measure_2.denoize_name, measure_1.rate, measure_1.model);
                    fprintf('Image_name_1:%s, PSNR:%f, SSIM:%f, MSE:%f \n',  measure.Image_name_1,  csnr(rec_im_1, measure_1.ori_im, 8,0, 0), cal_ssim(rec_im_1, measure_1.ori_im, 0, 0),MSE_im_1);
                    fprintf('Image_name_2:%s, PSNR:%f, SSIM:%f, MSE:%f \n',  measure.Image_name_2,  csnr(rec_im_2, measure_2.ori_im, 8,0, 0), cal_ssim(rec_im_2, measure_2.ori_im, 0, 0),MSE_im_2);
                    
                    fp = fopen(['../results/', Test_set{set}, '/', measure_1.denoize_name,'+',measure_2.denoize_name,'_multi.csv'],'a');
                    fprintf(fp,'%s, %s, %f, %f,%f,%f\n', measure.Image_name_1, measure_1.denoize_name,measure_1.rate,csnr(rec_im_1, measure_1.ori_im, 8,0, 0), cal_ssim(rec_im_1, measure_1.ori_im, 0, 0),MSE_im_1);
                    fprintf(fp,'%s, %s, %f, %f,%f,%f \n', measure.Image_name_2, measure_2.denoize_name,measure_2.rate,csnr(rec_im_2, measure_2.ori_im, 8,0, 0), cal_ssim(rec_im_2, measure_2.ori_im, 0, 0),MSE_im_2);
                    fclose(fp);

                end
                PSNR_eve = PSNR_sum/img_num;
                SSIM_eve = SSIM_sum/img_num;
                MSE_eve = MSE_sum/img_num;
                fp = fopen(['../results/', Test_set{set}, '/', measure_1.denoize_name,'+',measure_2.denoize_name,'average','_multi.csv'],'a');   %测试集所有图片在不同bpp下的平均R-D
                fprintf(fp,'%s, %s, %f,%f,%f,%f\n', measure.Image_name_1, measure_1.denoize_name,measure_2.rate,PSNR_eve, SSIM_eve,MSE_eve);
                fclose(fp);
            end

        end

    end

end
