function [x_hat,measure,PSNR] = Prox_Moment_Whole_analyze_curve(y_all, AMP_iters, denoiser, measure, quantize, par, PSNR_func)

y_all = cell2mat(y_all)';
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure.image_height,measure.image_width,denoiser);

OMEGA=cell2mat(measure.OMEGA(1:quantize.layer));
quantize.OMEGA = measure.OMEGA(1:quantize.layer);
k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
Phi_mt = measure.Phi./(k/measure.block_size^2);
M=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,measure.Phi);
Mt=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.image_height,measure.image_width,measure.block_height,measure.block_width,Phi_mt);


m=length(OMEGA);
if iscell(y_all)
    y=cell2mat(y_all)';
else
    y = y_all;
end

%%
n=measure.length;
m=length(y);
x_t{2} = zeros(n,1);
alpha = 1;
%%
tic;
v_t=Mt((M(x_t{2}))'-y);
x_t{1}=x_t{2}-alpha.*v_t;

[sigma_hat1,~] = NoiseLevel(reshape(x_t{1},measure.image_height,measure.image_width));
if sigma_hat1>90
    %sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},measure.image_height,measure.image_width));
    [sigma_hat1,~] = NoiseLevel(reshape(x_t{1},measure.image_height,measure.image_width));
end
x_t{2}=double(denoi(x_t{1},sigma_hat1));

%%


v_t = zeros(n,1);
PSNR=zeros(1,AMP_iters(1));
bit_errorNUM=zeros(AMP_iters(1)+1,3);
%[bit_errorNUM(1,1), bit_errorNUM(1,2)] = biterr(par.GroundTruth_symble{1}, par.bin{1});
%bit_errorNUM(1,3) = 1/m*sum((par.GroundTruth_symble{1}-double(par.bin{1})).^2);

x_hat = x_t{2};
i = 1;
measure.x_curve = zeros(1,AMP_iters);
measure.y_curve = zeros(1,AMP_iters);
measure.w_curve = zeros(1,AMP_iters);
measure.y_w_curve = zeros(1,AMP_iters);
x_temp{1} = x_t{2};
y_temp{1} = y;
w_temp{1} = cell2mat(par.dec)';



while i<=AMP_iters
    eta=randn(1,n)/sqrt(m);
    epsilon = 1;
    gamma=1/(epsilon).*eta*(denoi(x_t{1}+epsilon*eta',sigma_hat1)-x_t{2});
    %gamma = 0;
    v_t=gamma.*v_t+Mt((M(x_t{2}))'-y);
    x_t{1}=x_t{2}-alpha.*v_t;
    %sigma_hat1=SigEstmate_SigCNN(reshape(x_t{1},measure.image_height,measure.image_width));
    [sigma_hat1,~] = NoiseLevel(reshape(x_t{1},measure.image_height,measure.image_width));
    x_t{2}=double(denoi(x_t{1},sigma_hat1));
    PSNR(i) = PSNR_func(x_t{2});
    %%
    par.rim=x_t{2};
    [par]=whole_quantize(par,M, quantize);
    lambda_1 = 100;
    if iscell(par.dec)
        y=(M(x_t{2})'+lambda_1*cell2mat(par.dec)')/(1+lambda_1);
        %y=cell2mat(par.dec)';
    end
    
    
    
    x_temp{i+1} = x_t{2};
    y_temp{i+1} = y;
    w_temp{i+1} = cell2mat(par.dec)';
    
    measure.x_curve(1,i) = Rmse(x_temp{i+1},x_temp{i});
    measure.y_curve(1,i) = Rmse(y_temp{i+1},y_temp{i});
    measure.w_curve(1,i) = Rmse(w_temp{i+1},w_temp{i});
    measure.y_w_curve(1,i) = Rmse(w_temp{i+1},y_temp{i+1});
    
    %fprintf('x-x_ori: %.3f \n', sqrt(sum(sum((abs(measure.ori_im)-reshape(abs(x_t{2}),[measure.image_height,measure.image_width])).^2))/(measure.image_height*measure.image_width)));
    %fprintf('y-y_ori: %.3f \n', sqrt(sum((y-Trans.y{1}').^2)/length(y)));
    %fprintf('w-y_ori: %.3f \n', sqrt(sum((cell2mat(par.dec)'-Trans.y{1}').^2)/length(cell2mat(par.dec)')));
    %fprintf('y-w: %.3f \n', sqrt(sum((y-cell2mat(par.dec)').^2)/length(y)));
    
    
    i = i+1;
    %[bit_errorNUM(i,1), bit_errorNUM(i,2)] = biterr(int2bit(par.GroundTruth_symble{1},8), int2bit(par.bin{1},8));
    %bit_errorNUM(i,3) = 1/m*sum((par.GroundTruth_symble{1}-double(par.bin{1})).^2);
end
x_hat = x_t{2};
x_hat=reshape(x_hat,[measure.image_height,measure.image_width]);
dlmwrite(['../results/',[num2str(AMP_iters-1),'_PSNR_curve.csv']],PSNR,'-append');
elapsedTime = toc;
fprintf('Times:%f \n', elapsedTime);
fprintf('psnr:%f \n', PSNR(AMP_iters));
fprintf('iteration:%d \n', i);


end

function rmse = Rmse(y_true, y_pred)

error_squared = (y_true - y_pred).^2;

% 计算均方根误差
rmse = sqrt(mean(error_squared));


end


