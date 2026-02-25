function [x_hat,PSNR] = Prox(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

M=@(x) M_func(x);
Mt=@(z) Mt_func(z);
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);
%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
alpha = 1;
%%
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;
[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
x_t{2}=double(denoi(x_t{1},sigma_hat1));

%      [sigma_hat2,~] = NoiseLevel(reshape(x_t{2},height,width));
%      sigma_summary = [sigma_hat1,sigma_hat2];
%     error_summary = [norm(x_t{1}-measure.ori_im(:), 2)./n,norm(x_t{2}-measure.ori_im(:), 2)./n];

%%
%x_t{2} = extend_denoise(denoi, reshape(x_t{1},width,height), sigma_hat1);
PSNR=zeros(1,iters);
for i=1:iters
    
    if mod(i,4)==0
        alpha = alpha/1.8;
    end
    
    PSNR(i) = PSNR_func(x_t{2});
    v_t=Mt((M(x_t{2}))'-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    [sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    
%   x_t{2} = extend_denoise(denoi, reshape(x_t{1}, [width,height]), sigma_hat1);
%         [sigma_hat2,~] = NoiseLevel(reshape(x_t{2},height,width));
%          sigma_summary = [sigma_summary,sigma_hat1,sigma_hat2];
%          error_summary = [error_summary, norm(x_t{1}-measure.ori_im(:), 2)./n,norm(x_t{2}-measure.ori_im(:), 2)./n];
    
    
    
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


