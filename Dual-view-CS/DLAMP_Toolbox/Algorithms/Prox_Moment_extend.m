function [x_hat,PSNR] = Prox_Moment_extend(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

range_list = [0,5;5,10;10,15;15,20;20,30;30,40;40,50;50,60;60,70;70,80;80,90;90,100;100,125;125,150;150,300;300,500;500,1000];

global global_time
error_summary = [];
tar_range= [];
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
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;
error_temp = norm(measure.ori_im(:)-x_t{1})./length(x_t{1});
error_summary = [error_summary, error_temp];
global_time = global_time+toc;
[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
if sigma_hat1>90
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
end
tar_range = [tar_range,region_search(sigma_hat1, range_list)];
%x_t{2}=double(denoi(x_t{1},sigma_hat1));
x_t{2} = extend_denoise(denoi, reshape(x_t{1},width,height), sigma_hat1);
error_temp = norm(measure.ori_im(:)-x_t{2})./length(x_t{2});
error_summary = [error_summary, error_temp];




%%
v_t = zeros(n,1);
PSNR=zeros(1,iters);
for i=1:iters
    if mod(i,3)==0
        alpha = alpha/1;
    end
    PSNR(i) = PSNR_func(x_t{2});
    eta=randn(1,n);
    epsilon = 1;
    gamma=1/(m*epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    %gamma = 0.6;
    tic
    v_t=gamma.*v_t+Mt((M(x_t{2}))'-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    error_temp = norm(measure.ori_im(:)-x_t{1})./length(x_t{1});
    error_summary = [error_summary, error_temp];
    global_time = global_time+toc;
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    tar_range = [tar_range,region_search(sigma_hat1, range_list)];
    %x_t{2}=double(denoi(x_t{1},sigma_hat1));
    x_t{2} = extend_denoise(denoi, reshape(x_t{1}, [width,height]), sigma_hat1);
    error_temp = norm(measure.ori_im(:)-x_t{2})./length(x_t{2});
    error_summary = [error_summary, error_temp];
    
end

x_hat = x_t{2};


if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
    Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
    x_hat = x_hat+Mt(y-(M(x_hat))');
    PSNR(i+1) = PSNR_func(x_hat);
end

x_hat=reshape(x_hat,[height width]);
error_temp = norm(measure.ori_im(:)-x_hat(:))./length(x_hat);
error_summary = [error_summary, error_temp];

% dlmwrite(['../results/sigma/sigma_summary.csv'],sigma_summary,'-append');
%dlmwrite(['../results/sigma/error_summary_',num2str(measure.rate),'.csv'],error_summary,'-append');
%dlmwrite(['../results/sigma/range_summary_',num2str(measure.rate),'.csv'],tar_range ,'-append');


end


