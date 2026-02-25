%%
clear,close all;

add_path()
%%
randn('state',0);
rand('state',0);

SR = [30];
measure.iterative_way = 1;
%%
Test_set = { 'Set/Set11','urban100','Set/Set8','Set/Set11','Waterloo_crop/animal','Waterloo_crop/cityscape','Waterloo_crop/human'...
    'Waterloo_crop/landscape','Waterloo_crop/plant','Waterloo_crop/still_life','Waterloo_crop/transportation','Set/little5','BSD68', 'Set/Set12'};

for denoize_choice = [26] % [3,4,8,19,26][10:BM3D]
    for set = 1
        Test_image_dir = ['../../Test_Images/',Test_set{set},'/'];
        foldname = dir(Test_image_dir);
        foldname = foldname(3:end);
        img_num = length(foldname(:,1));
        measure.Test_set_name = Test_set{set};

        
        for kk= 1
            PSNR_sum = 0;
            SSIM_sum = 0;
            MSE_sum = 0;
            for k = 1:int32(img_num)
                measure.rate = 0.01*SR(kk);
                %% load original images
                k_1 = k;
                k_2 = k_1+1;

                if k_2 > img_num
                    k_2 = 1;
                end
                
                total_pixels = 0;
                img1 = imread(fullfile(Test_image_dir, foldname(k_1).name));
                img2 = imread(fullfile(Test_image_dir, foldname(k_2).name));
                total_pixels = numel(img1) + numel(img2);
                
                measure.rate = SR(kk) * 1000 / total_pixels; %固定测量
                
                measure.Image_name_1=foldname(k_1).name;
                measure.Image_name_2=foldname(k_2).name;
                
                measure.sigma_test = 1; 
                seed_1 = 1;
                seed_2 = 2;
                %%
                Test_image_dir_1 = fullfile(Test_image_dir, measure.Image_name_1);
                [measure_1,~] = para_set_individual(measure, denoize_choice, Test_image_dir_1, seed_1);

                Test_image_dir_2 = fullfile(Test_image_dir, measure.Image_name_2);
                [measure_2,~] = para_set_individual(measure_1, denoize_choice, Test_image_dir_2, seed_2);

                [measure_1, measure_2] = para_set_twoview_mix(measure, measure_1, measure_2);

        
               
                [rec_im_1,rec_im_2,MSE_im_1,MSE_im_2] = Enc_main(measure_1.ori_im, measure_2.ori_im, measure_1, measure_2);
                
                %add
%                 figure;
%                 subplot(1, 2, 1);
%                 imshow(rec_im_1, []);
%                 title(['Reconstructed Image 1: ', measure.Image_name_1]);
% 
%                 subplot(1, 2, 2);
%                 imshow(rec_im_2, []);
%                 title(['Reconstructed Image 2: ', measure.Image_name_2]);
                %add
                
                PSNR_sum = PSNR_sum+csnr(rec_im_1,  measure_1.ori_im, 8,0, 0)+csnr(rec_im_2,  measure_2.ori_im, 8,0, 0);
                SSIM_sum = SSIM_sum+cal_ssim(rec_im_1, measure_1.ori_im, 0, 0)+cal_ssim(rec_im_2, measure_2.ori_im, 0, 0);
                MSE_sum = MSE_sum+MSE_im_1+MSE_im_2;
                fprintf('Denoiser:%s, Rate:%f, measure_model:%s \n',measure_1.denoize_name, measure_1.rate, measure_1.model);
                fprintf('Image_name_1:%s, PSNR:%f, SSIM:%f, MSE:%f \n',  measure.Image_name_1,  csnr(rec_im_1, measure_1.ori_im, 8,0, 0), cal_ssim(rec_im_1, measure_1.ori_im, 0, 0),MSE_im_1);
                fprintf('Image_name_2:%s, PSNR:%f, SSIM:%f, MSE:%f \n',  measure.Image_name_2,  csnr(rec_im_2, measure_2.ori_im, 8,0, 0), cal_ssim(rec_im_2, measure_2.ori_im, 0, 0),MSE_im_2);
                
                fp = fopen(['../results/', Test_set{set}, '/', measure_1.denoize_name,'_multi.csv'],'a');
                fprintf(fp,'%s, %s, %f, %f,%f,%f\n', measure.Image_name_1, measure_1.denoize_name,measure_1.rate,csnr(rec_im_1, measure_1.ori_im, 8,0, 0), cal_ssim(rec_im_1, measure_1.ori_im, 0, 0),MSE_im_1);
                fprintf(fp,'%s, %s, %f, %f,%f,%f \n', measure.Image_name_2, measure_2.denoize_name,measure_2.rate,csnr(rec_im_2, measure_2.ori_im, 8,0, 0), cal_ssim(rec_im_2, measure_2.ori_im, 0, 0),MSE_im_2);
                fclose(fp);

            end
            PSNR_eve = PSNR_sum/img_num/2;
            SSIM_eve = SSIM_sum/img_num/2;
            MSE_eve = MSE_sum/img_num/2;
            fp = fopen(['../results/', Test_set{set}, '/', measure_1.denoize_name,'_average','_multi.csv'],'a');   %测试集所有图片在不同bpp下的平均R-D
            fprintf(fp,'%s, %s, %f,%f,%f,%f\n', measure.Image_name_1, measure_1.denoize_name,measure_2.rate,PSNR_eve, SSIM_eve,MSE_eve);
            fclose(fp);
        end

    end

end
