function [x_hat,PSNR] = Prox_Moment_Progressive(y_all, AMP_iters, denoiser, measure, quantize, par, PSNR_func)
global global_time

width =measure.image_height;
height = measure.image_width;
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

%% 先用基础层进行一次重建得到x_hat（x_t{2}），随后用多层一起联合重建，迭代中的y相当于四层拼接

y = cell2mat(y_all(1))'; %基础层
OMEGA=cell2mat(measure.OMEGA(1));
k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
Phi_mt = measure.Phi./(k/measure.block_size^2);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,Phi_mt);

%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
alpha = 1;
%% 恢复并去噪
tic
v_t=Mt((M(x_t{2}))'-y); %对基础层梯度下降
x_t{1}=x_t{2}-alpha.*v_t;
global_time = global_time+toc;
[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},height,width));
if sigma_hat1>90
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
end
x_t{2}=double(denoi(x_t{1},sigma_hat1)); %近端算子，去噪

%%
OMEGA=cell2mat(measure.OMEGA(1:quantize.layer));
quantize.OMEGA = measure.OMEGA(1:quantize.layer);
k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
Phi_mt = measure.Phi./(k/measure.block_size^2);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,Phi_mt);
m=length(OMEGA);
par.rim=x_t{2}; %x^
[par]=progressive_quantize(par,M, quantize); %par.dec为更新后的反量化后的细化层,步骤(x^压缩,量化,更新,反量化)
if iscell(par.dec)
    y=cell2mat(par.dec)';      %y代表的基础层被替换为多层
end
%% begin
v_t = zeros(n,1);
PSNR_temp = 0;
PSNR=zeros(1,AMP_iters(2));
x_hat = x_t{2};
i = 1;
while i<=AMP_iters(2) %增加迭代次数，20,AMP_iters(quantize.layer)
    eta=randn(1,n);
    epsilon = 1;
    gamma=1/(m*epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    tic
    v_t=gamma.*v_t+Mt((M(x_t{2}))'-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    global_time = global_time+toc;
    sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    PSNR(i) = PSNR_func(x_t{2});
    %%
    par.rim=x_t{2};
    tic
    [par]=progressive_quantize(par,M, quantize);
    global_time = global_time+toc;
    if iscell(par.dec)
        y = cell2mat(par.dec)';
        % y = (5*cell2mat(par.dec)'+(cell2mat(par.y))')./(5+1);
    end
    %%
    %         a(i,1) = max(x_t{2}(:));
    %         a(i,2) = min(x_t{2}(:));
    %         if (a(i,1) > 255 || a(i,2) < 0) && i>=2
    %             x_hat = x_t{2};
    %             break;
    %         else
    %             i = i+1;
    %             x_hat = x_t{2};
    %         end
    if PSNR(i)>=PSNR_temp
        x_hat = x_t{2};
        PSNR_temp = PSNR(i);
    end
    
    % x_show=reshape(x_hat,[height width]);
    % figure(i),imshow(x_show,[])
    
    i = i+1;
end
running_time=global_time;
fprintf('time:%f \n', running_time);
fp = fopen(['../results/', 'time','.csv'],'a');
fprintf(fp,'%f \n', running_time);
fclose(fp);


x_hat=reshape(x_hat,[height width]);
fprintf('iteration:%d \n', i);
% 观测值纠错统计
% True_bit = Trans.bit;
% error_sum = 0;
% for i=1:quantize.layer
%   True_bin{i}=bit2int(True_bit{i},quantize.refine_layer(i));
%   False_bin{i}=bit2int(par.bit{i},quantize.refine_layer(i));
%   error_sum = error_sum+sum(abs(True_bin{i}-False_bin{i}));
% end

end


