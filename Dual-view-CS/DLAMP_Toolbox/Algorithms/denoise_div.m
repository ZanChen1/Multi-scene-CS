function [ denoised, output_partial ] = denoise_div(noisy, noisy_r, sigma_hat,height,width,denoiser)

noisy=reshape(noisy,[height,width]);
noisy_r=reshape(noisy_r,[height,width]);


switch denoiser
    
    
    case 'Restormer_Div'
        
        noisy = reshape(noisy,height,width);
        noisy = noisy';
        noisy = reshape(noisy,1,height*width);
        denoise_out = py.Restormer_denoise_matlab_div.denoiser(noisy, height, width, sigma_hat);
        output = double(denoise_out{1});
        output_partial = double(denoise_out{2});
        output = reshape(output,width,height);
        output = output';
        output_partial = reshape(output_partial,width,height);
        output_partial = output_partial';
        
    case 'Restormer_jvp'
        
        noisy = reshape(noisy,height,width);
        noisy = noisy';
        noisy = reshape(noisy,1,height*width);
        
        noisy_r = reshape(noisy_r,height,width);
        noisy_r = noisy_r';
        noisy_r = reshape(noisy_r,1,height*width);
        
        
        denoise_out = py.Restormer_denoise_matlab_jvp.denoiser(noisy, noisy_r, height, width, sigma_hat);
        output = double(denoise_out{1});
        output_partial = double(denoise_out{2});
        output = reshape(output,width,height);
        output = output';
        output_partial = reshape(output_partial,width,height);
        output_partial = output_partial';
        
        
        
    case 'Restormer'
        
        if sigma_hat>=150
            output=denoise_RCAN_3x3_dilated_17cases(noisy, sigma_hat);
            %output=denoise_DnCNN_20Layers_3x3_17cases(noisy, sigma_hat);
        else
            noisy = reshape(noisy,height,width);
            noisy = noisy';
            noisy = reshape(noisy,1,height*width);
            output = double(py.Restormer_denoise_matlab.denoiser(noisy, height, width, sigma_hat));
            output = reshape(output,width,height);
            output = output';
        end
    otherwise
        error('Unrecognized Denoiser')
end
denoised=output(:);
output_partial = output_partial(:);
end

