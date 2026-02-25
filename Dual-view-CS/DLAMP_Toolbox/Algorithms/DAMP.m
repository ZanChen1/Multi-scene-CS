function [x_hat,PSNR_sum] = DAMP(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)
% function [x_hat,PSNR] = DAMP(y,iters,width,height,denoiser,M_func,Mt_func,PSNR_func)
% this function implements D-AMP based on any denoiser present in the
% denoise function
%
% Required Input:
%       y       : the measurements
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampeled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g., 'BM3D'
%       M_func  : function handle that projects onto M. Or a matrix M.
%
% Optional Input:
%       Mt_func  : function handle that projects onto M'.
%       PSNR_func: function handle to evaluate PSNR
%
% Output:
%       x_hat   : the recovered signal.
%       PSNR    : the PSNR trajectory.

if (nargin>=7)&&(~isempty(Mt_func)) % function handles
    M=@(x) M_func(x);
    Mt=@(z) Mt_func(z);
else % explicit Matrix
    M=@(x)M_func*x;
    Mt=@(z)M_func'*z;
end

% if ((nargin<8)||isempty(PSNR_func) )% no PSNR trajectory
%     PSNR_func = @(x) nan;
% end

denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

n=width*height;
m=length(y);
z_t=y;
x_t = zeros(n,1);
sigma_summary = zeros(1,iters+1);
error_summary = [];
div = zeros(1,iters);
PSNR_sum=zeros(1,iters);
flag_1 = 0;
sigma_test=measure.sigma_test;
if  isfield(measure,'predict_im') && flag_1 == 1
    x_t = measure.predict_im(:);
    pseudo_data = x_t+Mt(y-(M(x_t))');
    [sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
    x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    pseudo_data=Mt(z_t)+x_t;
    %pseudo_data = abs(pseudo_data);
    %[sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
    if sigma_test==1
        sigma_hat=SigEstmate_SigCNN(reshape(pseudo_data,height,width));
    else
        %sigma_hat = sqrt(1/m*sum(abs(z_t).^2));
        sigma_hat = sqrt(1/n*sum(abs(Mt(z_t)).^2));
    end
    %%
    sigma_summary(1)=sigma_hat;
    %x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

randn('state',0);
rand('state',0);
x_ttemp = x_t;
for i=1:iters
    PSNR_sum(i) = PSNR_func(x_t);
    error_temp = norm(measure.ori_im(:)-x_t)./length(x_t);
    error_summary = [error_summary, error_temp];
    %     if i>1 && PSNR(i)<PSNR(i-1)
    %         x_t = x_ttemp;
    %         break
    
    %     end
    %     x_ttemp = x_t;
    eta=randn(1,n);
    epsilon=max(pseudo_data)/1000+eps;
    div(i)=eta*((denoi(pseudo_data+epsilon*eta',sigma_hat)-x_t)/epsilon);
    z_t=double(y-(M(x_t))+1/m.*z_t.*div(i));   %%%%%%%%%%%%%%%%%%%%%%%%
    pseudo_data=Mt(z_t)+x_t;
    %pseudo_data = abs(pseudo_data);
    %[sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
    if sigma_test==1
        sigma_hat=SigEstmate_SigCNN(reshape(pseudo_data,height,width));
    else
        %sigma_hat = sqrt(1/m*sum(abs(z_t).^2));
        sigma_hat = sqrt(1/n*sum(abs(Mt(z_t)).^2));
    end
    sigma_summary(i+1)=sigma_hat;
    x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
PSNR_sum(i+1) = PSNR_func(x_t);
error_temp = norm(measure.ori_im(:)-x_t)./length(x_t);
error_summary = [error_summary, error_temp];
% if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
%     Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
%     x_t = x_t+Mt(y-(M(x_t)));
%     PSNR_sum(i+1) = PSNR_func(x_t);
% end
x_hat=reshape(x_t,[height width]);
dlmwrite(['../results/',measure.Test_set_name,'/sigma_summary_',measure.denoize_name,'_',num2str(sigma_test),'.csv'],sigma_summary,'-append');
dlmwrite(['../results/',measure.Test_set_name,'/PSNR_summary_',measure.denoize_name,'_',num2str(sigma_test),'.csv'],PSNR_sum,'-append');
dlmwrite(['../results/',measure.Test_set_name,'/RMSE_summary_',measure.denoize_name,'_',num2str(sigma_test),'.csv'],error_summary,'-append');
end



