function [x_hat,PSNR] = Prox_Moment_Whole(y_all, AMP_iters, denoiser, measure, quantize, par, PSNR_func)

y_all = cell2mat(y_all)';
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure.image_height,measure.image_width,denoiser);

OMEGA=cell2mat(measure.OMEGA(1:quantize.layer));
quantize.OMEGA = measure.OMEGA(1:quantize.layer);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi_mt);

% M=measure.A;
% Mt=measure.At;

m=length(OMEGA);
if iscell(y_all)
    y=cell2mat(y_all)';
else
    y = y_all;
end

%%
n=measure.length;
m=length(y);
x_t{2} = zeros(n,1);
alpha = 1;
%%

%y = y*0;

v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;

[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},measure.image_height,measure.image_width));
if sigma_hat1>90
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},measure.image_height,measure.image_width));
    %sigma_hat2=SigEstmate_SigCNN_2(reshape(x_t{1},measure.image_height,measure.image_width));
end
x_t{2}=double(denoi(x_t{1},sigma_hat1));
%%


v_t = zeros(n,1);
PSNR=zeros(1,AMP_iters(1));
%bit_errorNUM=zeros(AMP_iters(1)+1,3);
%[bit_errorNUM(1,1), bit_errorNUM(1,2)] = biterr(par.GroundTruth_symble{1}, par.bin{1});
%bit_errorNUM(1,3) = 1/m*sum((par.GroundTruth_symble{1}-double(par.bin{1})).^2);
i = 1;
while i<=AMP_iters
    eta=randn(1,n)/sqrt(m);
    epsilon = 1;
    gamma=1/(epsilon)*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    v_t=gamma.*v_t+Mt((M(x_t{2}))'-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    %sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},measure.image_height,measure.image_width));%*
    %sigma_hat1=SigEstmate_SigCNN_2(reshape(x_t{1},measure.image_height,measure.image_width));
    [sigma_hat1,~] = NoiseLevel(reshape(x_t{1},measure.image_height,measure.image_width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    PSNR(i) = PSNR_func(x_t{2});
    %%
    par.rim=x_t{2};
    [par]=whole_quantize(par,M, quantize);
    
    if iscell(par.dec)
        lambda_1 = 100;
        y=(M(x_t{2})'+lambda_1*cell2mat(par.dec)')/(1+lambda_1);
        %y=cell2mat(par.dec)';
    end
    i = i+1;
%     [bit_errorNUM(i,1), bit_errorNUM(i,2)] = biterr(int2bit(par.GroundTruth_symble{4},5), int2bit(par.bin{4},5));
%     bit_errorNUM(i,3) = 1/m*sum((par.GroundTruth_symble{3}-double(par.bin{3})).^2);
%     bit_errorNUM(i,4) = 1/m*sum((par.GroundTruth_symble{4}-double(par.bin{4})).^2);
end
%dlmwrite(['../results/',[num2str(iter_error-1),'PSNR.csv']],PSNR,'-append');
x_hat = x_t{2};
x_hat=reshape(x_hat,[measure.image_height,measure.image_width]);
fprintf('psnr:%f \n', PSNR(AMP_iters));
fprintf('iteration:%d \n', i);


end


