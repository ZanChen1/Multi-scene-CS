function [x_hat,PSNR] = Prox_Moment_complex(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

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
v_t=Mt((M(x_t{2}))-y);
x_t{1}=x_t{2}-alpha.*v_t;
global_time = global_time+toc;
[sigma_hat1,~] = NoiseLevel(reshape(abs(x_t{1}),height,width));
if sigma_hat1>90
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    sigma_hat1=SigEstmate_SigCNN(reshape(abs(x_t{1}),height,width));
end
sigma_hat = sqrt(1/m*sum(abs((M(x_t{2}))-y).^2));
x_t{2}=double(denoi(abs(x_t{1}),sigma_hat1));
x_t{2} = x_t{1}./abs(x_t{1}).*abs(x_t{2});
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
    v_t=gamma.*v_t+Mt((M(x_t{2}))-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    global_time = global_time+toc;
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    %sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    sigma_hat = sqrt(1/m*sum(abs((M(x_t{2}))'-y).^2));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    x_t{2} = x_t{1}./abs(x_t{1}).*abs(x_t{2});
    %x_t{2} = extend_denoise(denoi, reshape(x_t{1}, [width,height]), sigma_hat1);
    
end

x_hat = x_t{2};


if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
    Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
    x_hat = x_hat+Mt(y-(M(x_hat))');
    PSNR(i+1) = PSNR_func(x_hat);
end




x_hat=reshape(x_hat,[height width]);


% dlmwrite(['../results/sigma_summary.csv'],sigma_summary,'-append');
% dlmwrite(['../results/error_summary.csv'],error_summary,'-append');


end


