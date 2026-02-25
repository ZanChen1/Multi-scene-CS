function [x_hat,PSNR] = DAMP_2(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

if (nargin>=7)&&(~isempty(Mt_func)) % function handles
    M=@(x) M_func(x);
    Mt=@(z) Mt_func(z);
else % explicit Matrix
    M=@(x)M_func*x;
    Mt=@(z)M_func'*z;
end


denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

n=width*height;
m=length(y);
z_t=y;
x_t = zeros(n,1);
flag_1 = 0;

pseudo_data=Mt(z_t)+x_t;
[sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
x_t=double(denoi(pseudo_data,sigma_hat));   

 
Mp=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi);  

for i=1:iters
    
    PSNR(i) = PSNR_func(x_t);
    eta=randn(1,n);
    epsilon=max(pseudo_data)/1000+eps;
    div(i)=eta*((denoi(pseudo_data+epsilon*eta',sigma_hat)-x_t)/epsilon);
    %z_t=double(y-(M(x_t))'+1/m.*M(Mp(z_t))'.*div(i));   %%%%%%%%%%%%%%%%%%%%%%%%
    z_t=double(y-(M(x_t))'+1/m.*z_t.*div(i));   %%%%%%%%%%%%%%%%%%%%%%%%
    pseudo_data=Mt(z_t)+x_t;
    [sigma_hat,~] = NoiseLevel(reshape(pseudo_data,height,width));
    %sigma_hat = sqrt(1/m*sum(abs(z_t).^2));
    x_t=double(denoi(pseudo_data,sigma_hat));   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


Mp=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);    
x_t = x_t+Mp(y-(M(x_t))');
PSNR(i+1) = PSNR_func(x_t);

x_hat=reshape(x_t,[height width]);
end
