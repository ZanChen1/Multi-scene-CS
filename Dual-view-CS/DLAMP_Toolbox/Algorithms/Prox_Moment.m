function [x_hat,PSNR] = Prox_Moment(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

global global_time

M=@(x) M_func(x);

Mt=@(z) Mt_func(z);
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);
%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
alpha = 1;
%%
tic
v_t=Mt((M(x_t{2}))'-y); %y取负并重构
x_t{1}=x_t{2}-alpha.*v_t; %取正
global_time = global_time+toc;
[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width)); %对噪声分级
if sigma_hat1>90
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
end

x_t{2}=double(denoi(x_t{1},sigma_hat1)); %用去噪器去噪,x_t{2}已有初步原图形状，但很差

%x_t{2} = extend_denoise(denoi, reshape(x_t{1},width,height), sigma_hat1);



%%
v_t = zeros(n,1);
PSNR=zeros(1,iters);
for i=1:iters
    
    PSNR(i) = PSNR_func(x_t{2});
    eta=randn(1,n);
    epsilon = 1;
    gamma=1/(m*epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    tic
    v_t=gamma.*v_t+Mt((M(x_t{2}))'-y); %gamma+最初的重构
    x_t{1}=x_t{2}-alpha.*v_t; %取正
    global_time = global_time+toc;
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width)); %重复最初的步骤，迭代10次
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    %x_t{2} = extend_denoise(denoi, reshape(x_t{1}, [width,height]), sigma_hat1);
    
end

x_hat = x_t{2}; %有原图形状，但扭曲

%% 加上这部分后除了Barbara和Monarch，其他都变差，范围0-0.2dB
if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
    Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
    x_hat = x_hat+Mt(y-(M(x_hat))'); %重构+补充部分
    PSNR(i+1) = PSNR_func(x_hat);
end

x_hat=reshape(x_hat,[height width]);

% dlmwrite(['../results/sigma_summary.csv'],sigma_summary,'-append');
% dlmwrite(['../results/error_summary.csv'],error_summary,'-append');
end


