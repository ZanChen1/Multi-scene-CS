function [x_hat,PSNR_sum] = Prox_Moment_Bernoulli(y,iters,width,height,denoiser,M_func,Mt_func,measure,PSNR_func)

randn('state',0);
rand('state',0);

M=@(x) M_func(x);
Mt=@(z) Mt_func(z);

denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,height,width,denoiser);
%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
%x_t{2} = randn(n,1);
alpha = 1;
sigma_summary=zeros(2,iters+1);
%%
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;
%%
%[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
sigma_test=measure.sigma_test;
if sigma_test==1
    %sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    %sigma_hat1 = sqrt(1/n*sum(abs(v_t).^2));
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    %     x_temp = reshape(x_t{1},1,height*width);
    %     sigma_hat1 = double(py.Sigma_hat.sigmahalf(x_temp));
else
    %sigma_hat1 = sqrt(1/m*sum(abs(M(v_t)).^2)); %for Hardamard measurement
    sigma_hat1 = sqrt(1/n*sum(abs(v_t).^2));
end
%%
% sigma_hat0=std(measure.ori_im(:)-x_t{1});
% sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
% noisy = reshape(x_t{1},1,height*width);
% sigma_hat2 = double(py.Sigma_hat.sigmahalf(noisy));
% [sigma_hat3,~] = NoiseLevel(reshape(x_t{1},height,width));
% sigma_hat4 = sqrt(1/n*sum(abs(v_t).^2));
% sigma_summary(1,1)=sigma_hat0;
% sigma_summary(2,1)=sigma_hat1;
% sigma_summary(3,1)=sigma_hat2;
% sigma_summary(4,1)=sigma_hat3;
% sigma_summary(5,1)=sigma_hat4;

x_t{2}=double(denoi(x_t{1},sigma_hat1));


%Predicted_MSE_array(2)= DAMP_SE_Prediction(measure.ori_im(:), Predicted_MSE_array(1), m,n,0,denoiser,width,height);
%Predicted_sigma(1)=sqrt((n/m)*(Predicted_MSE_array(1)));

%%
v_t = zeros(n,1);
PSNR_sum=zeros(1,iters);
MSE_sum=zeros(1,iters);
for i=1:iters
    
    
    
    eta=randn(1,n);
    epsilon = 1;
    %sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    gamma=1/(m*epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    v_t=gamma.*v_t+Mt(y-(M(x_t{2}))')+mean(x_t{2});
    x_t{1}=x_t{2}+alpha.*v_t;
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    %noisy = reshape(x_t{1},1,height*width);
    %sigma_hat1 = double(py.Sigma_hat.sigmahalf(noisy));    
    %[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
    %sigma_hat1 = sqrt(1/n*sum(abs(v_t).^2));  
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    [PSNR_sum(i),MSE_sum(i)] = PSNR_func(x_t{2});
    
%     sigma_hat0=std(measure.ori_im(:)-x_t{1});
%     sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
%     noisy = reshape(x_t{1},1,height*width);
%     sigma_hat2 = double(py.Sigma_hat.sigmahalf(noisy));    
%     [sigma_hat3,~] = NoiseLevel(reshape(x_t{1},height,width));
%     sigma_hat4 = sqrt(1/n*sum(abs(v_t).^2));    
%     sigma_summary(1,i)=sigma_hat0;
%     sigma_summary(2,i)=sigma_hat1;
%     sigma_summary(3,i)=sigma_hat2;
%     sigma_summary(4,i)=sigma_hat3;
%     sigma_summary(5,i)=sigma_hat4;
    
    %Predicted_MSE_array(i+1)= DAMP_SE_Prediction(measure.ori_im(:), Predicted_MSE_array(i), m,n,0,denoiser,width,height);
    %Predicted_sigma(i+1)=sqrt((n/m)*(Predicted_MSE_array(i+1)));
    
end


[PSNR_sum(i+1),MSE_sum(i+1)] = PSNR_func(x_t{2});

i = i+1;
% if isfield(measure,'Phi_mp')&&(~isempty(measure.Phi_mp))
%     Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
%     x_t = x_t+Mt(y-(M(x_t)));
%     PSNR_sum(i+1) = PSNR_func(x_t{2});
% end
x_hat=reshape(x_t{2},[height width]);
imshow(x_hat,[]);
% dlmwrite(['../results/sigma_summary_',num2str(sigma_test),'.csv'],sigma_summary,'-append');
% dlmwrite(['../results/PSNR_summary_',num2str(sigma_test),'.csv'],PSNR_sum,'-append');

%dlmwrite(['../results/',measure.Test_set_name,'/B_sigma_summary_',measure.denoize_name,'_',num2str(sigma_test),'.csv'],sigma_summary,'-append');
%dlmwrite(['../results/',measure.Test_set_name,'/B_PSNR_summary_',measure.denoize_name,'_',num2str(sigma_test),'.csv'],PSNR_sum,'-append');
%dlmwrite(['../results/',measure.Test_set_name,'/B_RMSE_summary_',measure.denoize_name,'_',num2str(sigma_test),'.csv'],error_summary,'-append');

end


