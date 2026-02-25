function [x_hat,PSNR_sum] = DAMP_extend(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)
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

if measure.rate>=0.1
    Sig_measure = @(pseudo_data)NoiseLevel(reshape(pseudo_data,height,width));
else
    Sig_measure = @(pseudo_data)SigEstmate_SigCNN(reshape((pseudo_data),height,width));
end


% if ((nargin<8)||isempty(PSNR_func) )% no PSNR trajectory
%     PSNR_func = @(x) nan;
% end

denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

n=width*height;
m=length(y);
z_t=y;
x_t = zeros(n,1);
sigma_hat = zeros(1,iters);
div = zeros(1,iters);
PSNR_sum=zeros(1,iters);
flag_1 = 0;
if  isfield(measure,'predict_im') && flag_1 == 1
    x_t = measure.predict_im(:);
    pseudo_data = x_t+Mt(y-(M(x_t))');
    [sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
    %x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_t = extend_denoise(denoi, reshape(pseudo_data, [width,height]), sigma_hat);
else
    pseudo_data=Mt(z_t)+x_t;
    %pseudo_data = abs(pseudo_data);
    [sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
    %sigma_hat = sqrt(1/m*sum(abs(z_t).^2));
    %x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_t = extend_denoise(denoi, reshape(pseudo_data, [width,height]), sigma_hat);
end

randn('state',100);
rand('state',100);
x_ttemp = x_t;
for i=1:iters
    PSNR_sum(i) = PSNR_func(x_t);
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
    [sigma_hat] = Sig_measure(pseudo_data);
    %sigma_hat = sqrt(1/m*sum(abs(z_t).^2));
    %x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_t = extend_denoise(denoi, reshape(pseudo_data, [width,height]), sigma_hat);
end

if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
    Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
    x_t = x_t+Mt(y-(M(x_t)));
    PSNR_sum(i+1) = PSNR_func(x_t);
end
x_hat=reshape(x_t,[height width]);
end
