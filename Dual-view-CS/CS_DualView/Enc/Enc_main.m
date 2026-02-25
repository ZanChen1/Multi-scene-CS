function [predict_im_1, predict_im_2,MSE_im_1,MSE_im_2] = Enc_main(ori_im_1,ori_im_2, measure_1, measure_2);

%%

global global_time
%% measurement matrix
A_1 = measure_1.A;
At_1 = measure_1.At;
A_2 = measure_2.A;
At_2 = measure_2.At;

%% measure
y = A_1(ori_im_1)+A_2(ori_im_2);
measure = measure_1;        
%%
switch measure.denoize_name
    case 'NLR-CS'
        par = NLR_set_parameters(measure.rate);
        %At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
        tic
        [predict_im]   =  NLR_CS_Reconstruction( y, A, At, measure, par);
        global_time = global_time+toc;
        
    case 'TVNLR'
        theta = 2;
        beta = 5;
        opts_1 = opts_set(ori_im_1, measure.rate, theta, beta);
        opts_2 = opts_set(ori_im_2, measure.rate, theta, beta);
%         At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
        tic
        [predict_im_1, ~] = TVNLR(y',A_1,At_1,opts_1);
        [predict_im_2, ~] = TVNLR(y',A_2,At_2,opts_2);
        PSNR_func = @(x_hat, ori_im) PSNR(abs(ori_im),abs(x_hat));
        [~, MSE_im_1] = PSNR_func(predict_im_1, ori_im_1);
        [~, MSE_im_2] = PSNR_func(predict_im_2, ori_im_2);
        global_time = global_time+toc;
        
    case 'BCS-SPL'
        At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
        tic
        predict_im = BCS_SPL_DDWT_Decoder(y, measure.image_height,measure.image_width,5,200,A,At);
        global_time = global_time+toc;
        
    case 'RCAN_5x5_dilated_17cases+'
        %% AMP parameters setting
        AMP_iters = 10;
        errfxn = @(x_hat) PSNR(ori_im,reshape(x_hat,[measure.image_height,measure.image_width]));
        [predict_im,~]  = Prox_Moment_extend(y',AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
    case 'ADMM-Net'
        
        %addpath('../../ADMM-CSNet-master/Generic-ADMM-CSNet-Image/');
        %ADMM_net_config();
        predict_im = ADMM_Net(ori_im, measure);
        
        
    otherwise
        %% AMP parameters setting
        if measure.rate>0.45 
            AMP_iters = 10;
        elseif measure.rate>0.25 && isequal(measure.rate,0.25)
            AMP_iters = 15;
        else 
            AMP_iters = 20;
        end
        AMP_iters = 20;
        if strcmp(measure.model,'Diffraction')
            [predict_im,~]  = DAMP(y,AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
        else
            [predict_im_1,predict_im_2,~,MSE_sum]  = Prox_Moment_DualView(y',AMP_iters,A_1,At_1,A_2,At_2,measure_1,measure_2,measure.iterative_way);
            MSE_im_1 = MSE_sum(20,3);
            MSE_im_2 = MSE_sum(20,4);
            
        end

end


end