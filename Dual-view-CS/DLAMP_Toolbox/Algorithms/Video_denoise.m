function [ denoised ] = Video_denoise(noisy,sigma_hat,width,height,denoiser)


denoised=Video_denoise_RCAN_5x5_dilated_17cases(noisy, sigma_hat);
%denoised = Video_denoise_DnCNN_20Layers_3x3_17cases(noisy, sigma_hat);
end