function [x_hat,PSNR_sum] = Block_OAMP_alpha(y,iters,width,height,denoiser,measure,PSNR_func)


M = @(z)A_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi);
Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_pinv); %%%  Phi_pinv, Phi_lmmse, Phi_mt
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,measure.block_size,measure.block_size, denoiser);
%denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat, width, height, denoiser);
Block_denoi = @(noisy, sigma_hat)Block_denoiser(noisy, sigma_hat, measure.block_size, width,height, denoi);
%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
%% LE
r_t = Mt(y');
x_t{1}=x_t{2}+r_t;
%% NLE
sigma_hat = Block_SigEstmate(reshape(x_t{1},height,width), measure.block_size);
%sigma_hat=SigEstmate_SigCNN(reshape(x_t{1},height,width));
[x_t{2}, x_temp] = DivFree_denoiser2(x_t{1}, sigma_hat,measure.block_size, height, width, Block_denoi);
%%
PSNR_sum=zeros(1,iters);
MSE_sum=zeros(1,iters);
for i=1:iters
    %% LE
    r_t = Mt(y-(M(x_t{2}))');
    x_t{1}=x_t{2}+r_t;
    %% NLE
    sigma_hat = Block_SigEstmate(reshape(x_t{1},height,width), measure.block_size);
    %sigma_hat=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    [x_t{2}, x_temp] = DivFree_denoiser2(x_t{1}, sigma_hat,measure.block_size, height, width, Block_denoi);
    
    [PSNR_sum(i),MSE_sum(i)] = PSNR_func(x_temp);
end

x_hat=reshape(x_temp,[height width]);
[PSNR_sum(i),MSE_sum(i)] = PSNR_func(x_hat);
figure,imshow(x_hat,[])
end




function [x_3,x_2] = DivFree_denoiser2(x_1, sigma_hat, block_size, height, width, denoi)
n = size(x_1,1);
x_2=double(denoi(x_1,sigma_hat));
epsilon = 0.1;
for i =1 : 1
    eta=randn(1,n)/block_size;
    x_4 = denoi(x_1+eta'*epsilon,sigma_hat);
    
    x_4 = reshape(x_4, height, width);
    x_4 = im2col(x_4, [block_size block_size],'distinct');
    x_2 = reshape(x_2, height, width);
    x_2 = im2col(x_2, [block_size block_size],'distinct');
    x_1 = reshape(x_1, height, width);
    x_1 = im2col(x_1, [block_size block_size],'distinct');
    e = x_4-x_2;

    eta = reshape(eta, height, width);
    eta = im2col(eta, [block_size block_size],'distinct');
    Div_denoi = sum(e.*eta/epsilon, 1);
end
Div_denoi_mat = repmat(Div_denoi,[block_size*block_size,1]);
%Div_denoi_mean = mean(Div_denoi);
x_3 = (x_2 - x_1.* Div_denoi_mat)./(1-Div_denoi_mat);
x_3 = col2im(x_3,[block_size block_size],[height width],'distinct');
x_3 = abs(x_3(:));
x_2 = col2im(x_2,[block_size block_size],[height width],'distinct');
x_2 = abs(x_2(:));
end





function Div = Div_function(x, denoi, sigma_hat)

n = size(x,1);
eta=randn(1,n);
epsilon = 1;
x_1 = denoi(x,sigma_hat);
x_2 = denoi(x+eta'*epsilon,sigma_hat);
Div=1/n*eta*(x_2-x_1)/epsilon;

end


function sigma_hat = Block_SigEstmate(x, block_size)

B=im2col(x,[block_size block_size],'distinct');
for i=1:size(B,2)
    B_temp=reshape(B(:,i),[block_size,block_size]);
    sigma_hat(i)=SigEstmate_SigCNN(B_temp);
end

end


function x_out = Block_denoiser(x, sigma_hat, block_size, height, width, denoi)
x = reshape(x,height,width);
B=im2col(x,[block_size block_size],'distinct');
for i=1:size(B,2)
    B_temp=reshape(B(:,i),[block_size,block_size]);
    B_temp = denoi(B_temp, sigma_hat(i));
    B_output(:,i)=B_temp(:);
end
x_out=col2im(B_output,[block_size block_size],[height width],'distinct');
x_out = x_out(:);
end
