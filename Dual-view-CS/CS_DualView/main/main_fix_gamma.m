%%
clear,close all;
%%
% 是否需要用真实噪声值作为统计
%%
%%
add_path()
%%
randn('state',0);
rand('state',0);

SR = [20];
measure.iterative_way = 6;
%%
Test_set = {'Set/Set8','Set/Set11','Waterloo_crop/animal','Waterloo_crop/cityscape','Waterloo_crop/human'...
    'Waterloo_crop/landscape','Waterloo_crop/plant','Waterloo_crop/still_life','Waterloo_crop/transportation','Set/little5','BSD68', 'urban100', 'Set/Set12', 'Set/Set11'};

for denoize_choice = [26]%[4,19,22,23]  % [4,19,22,23]
    for set = 14 %[12,13]
        Test_image_dir = ['../../Test_Images/',Test_set{set},'/'];
        foldname = dir(Test_image_dir);
        foldname = foldname(3:end);
        img_num = length(foldname(:,1));
        measure.Test_set_name = Test_set{set};
        
        for gamma = [0,0.5,1]
            measure.gamma = gamma;
%             save('par_ablation_gamma.mat','gamma')
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
                    fprintf('Denoiser:%s, Rate:%f, measure_model:%s ,gamma=%f\n',measure_1.denoize_name, measure_1.rate, measure_1.model,gamma);
                    fprintf('Image_name_1:%s, PSNR:%f, SSIM:%f, MSE:%f \n',  measure.Image_name_1,  csnr(rec_im_1, measure_1.ori_im, 8,0, 0), cal_ssim(rec_im_1, measure_1.ori_im, 0, 0),MSE_im_1);
                    fprintf('Image_name_2:%s, PSNR:%f, SSIM:%f, MSE:%f \n',  measure.Image_name_2,  csnr(rec_im_2, measure_2.ori_im, 8,0, 0), cal_ssim(rec_im_2, measure_2.ori_im, 0, 0),MSE_im_2);

                    fp = fopen(['../results/', Test_set{set}, '/',num2str(gamma),'/', measure_1.denoize_name,'_multi_gamma.csv'],'a');
                    fprintf(fp,'%s, %s, %f, %f, %f,%f,%f\n', measure.Image_name_1, measure_1.denoize_name,gamma,measure_1.rate,csnr(rec_im_1, measure_1.ori_im, 8,0, 0), cal_ssim(rec_im_1, measure_1.ori_im, 0, 0),MSE_im_1);
                    fprintf(fp,'%s, %s, %f, %f, %f,%f,%f \n', measure.Image_name_2, measure_2.denoize_name,gamma,measure_2.rate,csnr(rec_im_2, measure_2.ori_im, 8,0, 0), cal_ssim(rec_im_2, measure_2.ori_im, 0, 0),MSE_im_2);
                    fclose(fp);

                end
                PSNR_eve = PSNR_sum/img_num/2;
                SSIM_eve = SSIM_sum/img_num/2;
                MSE_eve = MSE_sum/img_num/2;
                fp = fopen(['../results/', Test_set{set}, '/',num2str(gamma),'/', measure_1.denoize_name,'average','_multi_gamma.csv'],'a');   
                fprintf(fp,'%s, %s, %f, %f,%f,%f,%f\n', measure.Image_name_2, measure_1.denoize_name,gamma,measure_2.rate,PSNR_eve, SSIM_eve,MSE_eve);
                fclose(fp);

            end
            fp = strcat('../results/', Test_set{set}, '/',num2str(gamma),'/','ablation_',measure_1.denoize_name,'fix_gamma=',num2str(gamma),'.csv');
            PSNR_15iter = readmatrix(fp);
            sigma_maean = mean(PSNR_15iter);
            writematrix(sigma_maean,fp,'WriteMode','append');
        end

    end

end
