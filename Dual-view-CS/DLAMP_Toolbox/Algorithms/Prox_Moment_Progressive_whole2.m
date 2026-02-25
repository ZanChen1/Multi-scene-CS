function [x_hat,PSNR] = Prox_Moment_Progressive_whole2(y_all, AMP_iters, denoiser, measure, quantize, par, PSNR_func,lambda2)
rand('state',1);
randn('state',1);
width =measure.image_height;
height = measure.image_width;
Pvalue_dec = par.Pvalue_dec;
Pvalue_prob = par.Pvalue_prob;
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);
y = cell2mat(y_all(1))';
OMEGA=cell2mat(measure.OMEGA(1));
k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
Phi_mt = measure.Phi./(k/measure.block_size^2);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,Phi_mt);
%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
alpha = 1;
%%
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;
[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
if sigma_hat1>90
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
end
x_t{2}=double(denoi(x_t{1},sigma_hat1));
%%
OMEGA=cell2mat(measure.OMEGA(1:quantize.layer));
quantize.OMEGA = measure.OMEGA(1:quantize.layer);
k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
Phi_mt = measure.Phi./(k/measure.block_size^2);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,Phi_mt);
m=length(OMEGA);
par.rim=x_t{2};
y = M(x_t{2});
%%

% lambda1_min = 1e-4;   % 初始较软  for bit level = 5;
% lambda1_max = 1e-1;   % 最终较硬
% lambda1_min = 1e-4;   % 初始较软  for bit level = 7;
% lambda1_max = 0.5e-1;   % 最终较硬
lambda1_min = measure.lambda1_min;   % 初始较软  
lambda1_max = measure.lambda1_max;   % 最终较硬
lambda1 = lambda1_min;
y = whole_quantize_2(y, Pvalue_dec, Pvalue_prob,lambda1);
y_error(1) = sum(abs(y - (M(measure.ori_im))))./length(y);
%%
v_t = zeros(n,1);
PSNR=zeros(1,AMP_iters);
x_hat = x_t{2};
hat = zeros(1,AMP_iters);
i = 1;




while i<=AMP_iters
    lambda1_alpha = (i - 1) / max(AMP_iters - 1, 1);   % 归一化到 [0,1]
    lambda1= lambda1_min * (lambda1_max / lambda1_min)^lambda1_alpha;
    eta=randn(1,n);
    epsilon = 1;
    gamma=1/(m*epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    v_t=gamma.*v_t+Mt((M(x_t{2}))-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    %         [sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    hat(i) = sigma_hat1;
    PSNR(i) = PSNR_func(x_t{2});
    %     SSIM(i) =  cal_ssim(abs(measure.ori_im),reshape(abs(x_t{2}),[measure.image_height,measure.image_width]),0,0);
    % %     SSIM2(i) =  ssim((measure.ori_im),reshape((x_t{2}),[measure.image_height,measure.image_width]));
    %     MSE(i) = mse(abs(measure.ori_im)-reshape(abs(x_t{2}),[measure.image_height,measure.image_width]));


    %%
    par.rim=x_t{2};
    y = M(x_t{2});
    [y_whole]=whole_quantize_2(y, Pvalue_dec, Pvalue_prob, lambda1);
    y=(lambda2*y_whole+y)./(lambda2+1);
    y_error(i+1) = sum(abs(y - (M(measure.ori_im))))./length(y);
    x_hat = x_t{2};
    %
    %     if sigma_temp > hat(i)
    %         x_hat = x_t{2};
    %         sigma_temp = hat(i);
    %     end

    %     if PSNR(i)>=PSNR_temp
    %         x_hat = x_t{2};
    %         PSNR_temp = PSNR(i);
    %     end
    i = i+1;
end

% dlmwrite(['../results/',[num2str(iter_error-1),'PSNR.csv']],PSNR,'-append');
% dlmwrite(['../results/',[num2str(iter_error-1),'SSIM.csv']],SSIM,'-append');
% dlmwrite(['../results/',[num2str(iter_error-1),'MSE.csv']],MSE,'-append');

x_hat=reshape(x_hat,[height width]);
fprintf('iteration:%d \n', i);

end


