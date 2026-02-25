function [x_hat_1,x_hat_2,x_syn,PSNR_sum] = Prox_Moment_DualView(y,iters,M_func_1,Mt_func_1,M_func_2,Mt_func_2,measure_1,measure_2)

randn('state',0);
rand('state',0);

PSNR_func = @(x_hat, ori_im) PSNR(abs(ori_im),abs(x_hat));
M_1=@(x) M_func_1(x);
Mt_1=@(z) Mt_func_1(z);
M_2=@(x) M_func_2(x);
Mt_2=@(z) Mt_func_2(z);
denoi_1=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure_1.image_height,measure_1.image_width,measure_1.denoize_name);
denoi_2=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure_2.image_height,measure_2.image_width,measure_2.denoize_name);
%%
m=length(y);
n_1=measure_1.length;
n_2=measure_2.length;
mask_1 = measure_1.mask;
mask_2 = measure_2.mask;
mask_1_dilated = mask_1+measure_1.mask_dilated;
mask_2_dilated = mask_2+measure_2.mask_dilated;




n_1 = sum(mask_1(:));
n_2 = sum(mask_2(:));
x_t_1{2} = zeros(measure_1.length,1);
x_t_2{2} = zeros(measure_2.length,1);
alpha = 1;
%%
x_t_1{1}=Mt_1(y);
x_t_2{1}=Mt_2(y);
%%
%[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
%%
sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
sigma_hat_2=SigEstmate_SigCNN(reshape(x_t_2{1},measure_2.image_height,measure_2.image_width));
%sigma_hat = sqrt(sigma_hat_1*sigma_hat_2);
x_t_1{2}=double(denoi_1(x_t_1{1},sigma_hat_1)).*mask_1(:);
x_t_2{2}=double(denoi_2(x_t_2{1},sigma_hat_2)).*mask_2(:);

%%
v_t_1 = zeros(measure_1.length,1);
v_t_2 = zeros(measure_2.length,1);
PSNR_sum=zeros(iters,4);
MSE_sum=zeros(iters,4);
y_error = zeros(iters,1);
sigma_sum = zeros(iters,2);
iterative_way = 1;

eta_1=randn(1,measure_1.length).*mask_1(:)';
eta_2=randn(1,measure_2.length).*mask_2(:)';
epsilon = 1;
gamma_1=1/(m*epsilon).*eta_1*(denoi_1(x_t_1{1}+epsilon*eta_1',sigma_hat_1).*mask_1(:)-x_t_1{2});
gamma_2=1/(m*epsilon).*eta_2*(denoi_2(x_t_2{1}+epsilon*eta_2',sigma_hat_2).*mask_2(:)-x_t_2{2});

x_syn= x_t_1{1}.*mask_1(:)+x_t_2{1}.*mask_2(:);
x_syn_1 = x_syn.*mask_1_dilated(:)+x_t_1{1}.*(1-mask_1_dilated(:));
x_syn_2 = x_syn.*mask_2_dilated(:)+x_t_2{1}.*(1-mask_2_dilated(:));

for i=1:iters
    
    
    switch iterative_way
        case 1
            v_temp = y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))';
            gamma_1=1/(m*epsilon).*eta_1*(denoi_1(x_syn_1+epsilon*eta_1',sigma_hat_1).*mask_1(:)-x_t_1{2});
            gamma_2=1/(m*epsilon).*eta_2*(denoi_2(x_syn_2+epsilon*eta_2',sigma_hat_2).*mask_2(:)-x_t_2{2});
            gamma = gamma_1+gamma_2;
            
            v_t_1=gamma.*v_t_1+Mt_1(v_temp);
            x_t_1{1}=x_t_1{2}+alpha.*v_t_1;
            %sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
            %sigma_hat_1=std(measure_1.ori_im(:)-x_t_1{1});
            sigma_hat_1 = sqrt(norm(Mt_1(v_temp)).^2/n_1);
            if isnan(sigma_hat_1)
                sigma_hat_1 = 0;
            end
            v_t_2=gamma.*v_t_2+Mt_2(v_temp);
            x_t_2{1}=x_t_2{2}+alpha.*v_t_2;
            %sigma_hat_2=SigEstmate_SigCNN(reshape(x_t_2{1},measure_2.image_height,measure_2.image_width));
            %sigma_hat_2 = std(measure_2.ori_im(:)-x_t_2{1});
            sigma_hat_2 = sqrt(norm(Mt_2(v_temp)).^2/n_2);
            if isnan(sigma_hat_2)
                sigma_hat_2 = 0;
            end
            
            x_syn= x_t_1{1}.*mask_1(:)+x_t_2{1}.*mask_2(:);
            x_syn_1 = x_syn.*mask_1_dilated(:)+x_t_1{1}.*(1-mask_1_dilated(:));
            x_syn_2 = x_syn.*mask_2_dilated(:)+x_t_2{1}.*(1-mask_2_dilated(:));
            x_t_1{2}=double(denoi_1(x_syn_1,sigma_hat_1)).*mask_1(:);
            x_t_2{2}=double(denoi_2(x_syn_2,sigma_hat_2)).*mask_2(:);
            
        case 2
            
            
            
            
            
            
            gamma = gamma_1+gamma_2;
            v_temp = y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))';
            v_t_1=gamma.*v_t_1+Mt_1(v_temp);
            x_t_1{1}=x_t_1{2}+alpha.*v_t_1;
            sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
            x_t_1{2}=double(denoi_1(x_t_1{1},sigma_hat_1));
            gamma_1=1/(m*epsilon).*eta*(denoi_1(x_t_1{1}+epsilon*eta',sigma_hat_1)-x_t_1{2});
            
            
            
            
            gamma = gamma_1+gamma_2;
            v_temp = y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))';
            v_t_2=gamma.*v_t_2+Mt_2(v_temp);
            x_t_2{1}=x_t_2{2}+alpha.*v_t_2;
            sigma_hat_2=SigEstmate_SigCNN(reshape(x_t_2{1},measure_2.image_height,measure_2.image_width));
            x_t_2{2}=double(denoi_2(x_t_2{1},sigma_hat_2));
            gamma_2=1/(m*epsilon).*eta*(denoi_2(x_t_2{1}+epsilon*eta',sigma_hat_2)-x_t_2{2});
            
        case 3 %%单一图片的对比实验情况
            v_temp = y-(M_1(x_t_1{2}))';
            gamma=1/(m*epsilon).*eta_1*(denoi_1(x_t_1{1}+epsilon*eta_1',sigma_hat_1)-x_t_1{2});
            v_t_1=gamma.*v_t_1+Mt_1(v_temp);
            x_t_1{1}=x_t_1{2}+alpha.*v_t_1;
            sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
            x_t_1{2}=double(denoi_1(x_t_1{1},sigma_hat_1));
            
        case 4 %%OAMP dual view图片的对比实验情况
            v_temp = y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))';
            
            v_t_1=Mt_1(v_temp);
            x_t_1{1}=x_t_1{2}+alpha.*v_t_1;
            %sigma_hat_1=SigEstmate_SigCNN(reshape(x_t_1{1},measure_1.image_height,measure_1.image_width));
            sigma_hat_1 = sqrt(norm(Mt_1(v_temp)).^2/n_2);
            
            v_t_2=Mt_2(v_temp);
            x_t_2{1}=x_t_2{2}+alpha.*v_t_2;
            %sigma_hat_2=SigEstmate_SigCNN(reshape(x_t_2{1},measure_2.image_height,measure_2.image_width));
            sigma_hat_2 = sqrt(norm(Mt_2(v_temp)).^2/n_2);
            
            x_t_1{2}=double(denoi_1(x_t_1{1},sigma_hat_1)).*mask_1(:);
            x_t_2{2}=double(denoi_2(x_t_2{1},sigma_hat_2)).*mask_2(:);
            
            gamma_1=1/(n_1*epsilon).*eta_1*(denoi_1(x_t_1{1}+epsilon*eta_1',sigma_hat_1).*mask_1(:)-x_t_1{2});
            gamma_2=1/(n_2*epsilon).*eta_2*(denoi_2(x_t_2{1}+epsilon*eta_2',sigma_hat_2).*mask_2(:)-x_t_2{2});
            %gamma = gamma_1+gamma_2;
            x_t_1{2} = (x_t_1{2} - gamma_1*x_t_1{1})./(1-gamma_1);
            x_t_2{2} = (x_t_2{2} - gamma_2*x_t_2{1})./(1-gamma_2);
            
            
            
            
        otherwise
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    y_error(i) = sum(abs(v_temp)/m);
    sigma_sum(i,1) = sigma_hat_1;
    sigma_sum(i,2) = sigma_hat_2;
    I1 = reshape(x_t_1{1},measure_1.image_height,measure_1.image_width);
    I2 = reshape(x_t_2{1},measure_2.image_height,measure_2.image_width);
    I3 = reshape(x_t_1{2},measure_1.image_height,measure_1.image_width);
    I4 = reshape(x_t_2{2},measure_2.image_height,measure_2.image_width);
    [PSNR_sum(i,1), MSE_sum(i,1)] = PSNR_func(I1, measure_1.ori_im);
    [PSNR_sum(i,2), MSE_sum(i,2)] = PSNR_func(I2, measure_2.ori_im);
    [PSNR_sum(i,3), MSE_sum(i,3)] = PSNR_func(I3, measure_1.ori_im);
    [PSNR_sum(i,4), MSE_sum(i,4)] = PSNR_func(I4, measure_2.ori_im);
    
end


x_hat_1=reshape(x_t_1{2},[measure_1.image_height measure_1.image_width]);
x_hat_2=reshape(x_t_2{2},[measure_2.image_height measure_2.image_width]);
figure(1),imshow(x_hat_1,[]);
figure(2),imshow(x_hat_2,[]);
%x_syn = (x_hat_1.*mask_1+x_hat_2.*mask_2)./(mask_1+mask_2);
x_syn = (x_hat_1.*(exp(100*mask_1)-1)+x_hat_2.*(exp(100*mask_2)-1))./((exp(100*mask_1)-1)+(exp(100*mask_2)-1));
figure(3),imshow(x_syn,[]);
[PSNR_sum(i,3), MSE_sum(i,3)] = PSNR_func(x_hat_1, measure_1.ori_im);
[PSNR_sum(i,4), MSE_sum(i,4)] = PSNR_func(x_hat_2, measure_2.ori_im);
% if isfield(measure_1,'Phi_pinv')&&(~isempty(measure_1.Phi_pinv))
%     w_coef = measure_1.w_coef;
%     Mt_1=@(z)At_bp(z,measure_1.OMEGA,measure_1.P_image,measure_1.P_block,measure_1.image_height,measure_1.image_width,measure_1.block_height,measure_1.block_width,measure_1.Phi_pinv2,1/w_coef,measure_1.mask);
%     Mt_2=@(z)At_bp(z,measure_2.OMEGA,measure_2.P_image,measure_2.P_block,measure_2.image_height,measure_2.image_width,measure_2.block_height,measure_2.block_width,measure_2.Phi_pinv2,1/(1-w_coef),measure_2.mask);
%     x_hat_1 = x_syn(:).*mask_1(:)+Mt_1(y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))').*mask_1(:);
%     x_hat_2 = x_syn(:).*mask_2(:)+Mt_2(y-(M_1(x_t_1{2}))'-(M_2(x_t_2{2}))').*mask_2(:);
% end

% x_hat_1=reshape(x_hat_1,[measure_1.image_height measure_1.image_width]);
% x_hat_2=reshape(x_hat_2,[measure_2.image_height measure_2.image_width]);
%
%

% MSE_sum = MSE_sum(i,3)+MSE_sum(i,4);
% fprintf('MSE_sum_1:%s,  \n',measure_1.denoize_name, measure_1.rate, measure_1.model);
% x_syn = (x_hat_1.*(exp(1000*mask_1)-1)+x_hat_2.*(exp(1000*mask_2)-1))./((exp(1000*mask_1)-1)+(exp(1000*mask_2)-1));
%
% figure(4),imshow(x_hat_1,[]);
% figure(5),imshow(x_hat_2,[]);
% figure(6),imshow(x_syn_2,[]);





end


