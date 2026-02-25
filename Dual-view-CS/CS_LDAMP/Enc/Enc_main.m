function predict_im = Enc_main(ori_im, measure)

%%
addpath('../Utilities');
addpath('../Utilities/Measurements');
addpath('../Utilities/Measurements/mask');
global global_time
%% measurement matrix
[A, At, measure] = Measure_matrix_create(measure);

%% measure
measure.ori_im = ori_im;
y = A(ori_im);

%%
switch measure.denoize_name
    case 'NLR-CS'
        par = NLR_set_parameters(measure.rate);
        %At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
        tic
        [predict_im]   =  NLR_CS_Reconstruction1( y, A, At, measure, par);
        global_time = global_time+toc;
        
    case 'TVNLR'
        theta = 2;
        beta = 5;
        opts = opts_set(ori_im, measure.rate, theta, beta);
        At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
        tic
        [predict_im, ~] = TVNLR(y',A,At,opts);
        global_time = global_time+toc;
        
    case 'BCS-SPL'
        At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
        tic
        predict_im = BCS_SPL_DDWT_Decoder(y, measure.image_height,measure.image_width,5,200,A,At);
        global_time = global_time+toc;
        
    case 'RCAN_5x5_dilated_17cases+'
        %% AMP parameters setting
        AMP_iters = 15;
        errfxn = @(x_hat) PSNR(ori_im,reshape(x_hat,[measure.image_height,measure.image_width]));
        [predict_im,~]  = Prox_Moment_extend(y',AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
%         [predict_im,~]  = DAMP(y,AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
    case 'ADMM-Net'
        
        %addpath('../../ADMM-CSNet-master/Generic-ADMM-CSNet-Image/');
        %ADMM_net_config();
        predict_im = ADMM_Net(ori_im, measure);
    otherwise
        %% AMP parameters setting
        AMP_iters =20;
        errfxn = @(x_hat) PSNR(abs(ori_im),reshape(abs(x_hat),[measure.image_height,measure.image_width]));
        %[predict_im,~]  = Prox_Moment_test(y',AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
        %[predict_im,~]  = Prox_Moment_complex(y,AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
        if strcmp(measure.model,'Diffraction')
            [predict_im,~]  = DAMP(y,AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
        else
            [predict_im,~]  = Prox_Moment_alpha(y',AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
            %[predict_im,~]  = Prox_Moment_test(y',AMP_iters,measure.image_height,measure.image_width,measure.denoize_name,A,At,measure,errfxn);
        end
end


end