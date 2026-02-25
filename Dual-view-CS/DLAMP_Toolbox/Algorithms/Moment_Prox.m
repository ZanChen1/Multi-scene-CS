function [x_hat,PSNR] = Moment_Prox(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

M=@(x) M_func(x);
Mt=@(z) Mt_func(z);
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);
%%
n=width*height;
m = length(y);
x_t{2} = zeros(n,1);
alpha = 0.5;
%%
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;
[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
x_t{2}=double(denoi(x_t{1},sigma_hat1));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_t{3} = zeros(n,1);

iters = 10;
PSNR=zeros(1,iters);
for i=1:iters
        
    PSNR(i) = PSNR_func(x_t{2});
    eta=randn(1,n);
    epsilon = 1;
    gamma=1/(m*epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    v_t=Mt((M(x_t{2})'-y))+gamma.*(x_t{3}-x_t{1});  
    x_t{1}=x_t{2}-alpha.*v_t;
    x_t{3} = x_t{2};
    [sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
       
    
end

x_hat = x_t{2};

if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
    Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
    x_hat = x_hat+Mt(y-(M(x_hat))');
    PSNR(i+1) = PSNR_func(x_hat);
end

x_hat=reshape(x_hat,[height width]);




end



