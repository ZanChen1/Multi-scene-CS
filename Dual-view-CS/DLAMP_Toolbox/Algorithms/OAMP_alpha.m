function [x_hat,PSNR_sum] = OAMP_alpha(y,iters,width,height,denoiser,measure,PSNR_func)

randn('state',1);
rand('state',1);
M = @(z)A_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi);
Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_pinv); %%%  Phi_pinv, Phi_lmmse, Phi_mt
At=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mt); %%%  Phi_pinv, Phi_lmmse, Phi_mt
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,height,width,denoiser);

%%
n=width*height;
m=length(y);
x_t{2} = zeros(n,1);
%% LE
z_t = Mt(y');
x_t{1}=x_t{2}+z_t;
%% NLE
%sigma_hat=SigEstmate_SigCNN(reshape(x_t{1},height,width));
%sigma_hat = sqrt(norm(r_t,2).^2/n);
s_t{1} = x_t{2};
r_t{1} = x_t{1};
sigma_hat = sqrt(0.5*(norm(z_t,2).^2/n+norm(y'-M(x_t{1})).^2/n));
[x_t{2}, ~] = DivFree_denoiser1(x_t{1}, z_t, sigma_hat, denoi);

%[x_t{2}, x_temp] = DivFree_denoiser2(x_t{1}, sigma_hat, denoi);
%[x_t{2}, x_temp] = DivFree_denoiser5(x_t{1}, r_t, y, sigma_hat, M, Mt, denoi, measure.ori_im);
%%
PSNR_sum=zeros(1,iters);
MSE_sum=zeros(1,iters);
[block_m, block_n] = size(measure.Phi);
Phi2Phi = measure.Phi*measure.Phi';
sigma2_noise = 1;
for i=1:iters
    %% LE
    %     v_sigma2 = norm(y'-M(x_t{3}))^2/m-sigma2_noise;
    %     Phi_lmmse_hat = measure.Phi'/(Phi2Phi+sigma2_noise/v_sigma2*eye(block_m));
    %     Phi_lmmse = Phi_lmmse_hat*block_n/trace(Phi_lmmse_hat*measure.Phi*eye(block_n));
    %     measure.Phi_lmmse = Phi_lmmse;
    %     Mt=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_lmmse);

    z_t = Mt(y-(M(x_t{2}))');
    x_t{1}=x_t{2}+z_t;
    s_t{i+1} = x_t{2};
    r_t{i+1} = x_t{1};
    %% NLE
    %sigma_hat=SigEstmate_SigCNN(reshape(x_t{1},height,width));
    %sigma_hat = sqrt(norm(r_t,2).^2/n);
    sigma_hat = sqrt(0.5*(norm(z_t,2).^2/n+norm(y'-M(x_t{1})).^2/n));
    %[x_t{2}, x_temp] = DivFree_denoiser1(x_t{1}, r_t, sigma_hat, denoi);
    [x_t{2}, x_temp] = DivFree_denoiser2(x_t{1}, sigma_hat, denoi);
    q_t = x_t{2} - measure.ori_im(:);
    h_t = x_t{1} - measure.ori_im(:); 
    %[x_t{2}, x_temp] = DivFree_denoiser5(x_t{1}, z_t, s_t, r_t, y, sigma_hat, M, Mt, denoi, measure.ori_im, i);
    [PSNR_sum(i),MSE_sum(i)] = PSNR_func(x_temp);
    fprintf('Iter:%d, Image_name:%s, PSNR:%f, q_t var:%f, h_t var:%f \n', i, measure.Image_name,  PSNR_sum(i), std(q_t), std(h_t));
    var(q_t)/var(h_t)
    imshow(reshape(x_temp,[height width]),[])
end

x_hat=reshape(x_temp,[height width]);
[PSNR_sum(i),MSE_sum(i)] = PSNR_func(x_hat);
imshow(x_hat,[])
end


function [x_3,x_2] = DivFree_denoiser1(x_1, r_t, sigma_hat, denoi)
n = size(x_1,1);
x_2=denoi(x_1,sigma_hat);
epsilon = 0.1;
for i =1 : 1
    eta=randn(1,n);
    Div_denoi(i)=1*eta/n*(denoi(x_1+eta'*epsilon,sigma_hat)-x_2)/epsilon;
end
Div_denoi_mean = mean(Div_denoi);
x_3 = x_2 - r_t.* Div_denoi_mean;
end

function [x_3,x_2] = DivFree_denoiser2(x_1, sigma_hat, denoi)
n = size(x_1,1);
x_2=double(denoi(x_1,sigma_hat));
epsilon = 0.1;
for i =1 : 1
    eta=randn(1,n);
    Div_denoi(i)=1*eta/n*(denoi(x_1+eta'*epsilon,sigma_hat)-x_2)/epsilon;
end
Div_denoi_mean = mean(Div_denoi);
x_3 = (x_2 - x_1.* Div_denoi_mean)./(1-Div_denoi_mean);
end


function [x_3,x_2] = DivFree_denoiser3(x_1, sigma_hat, denoi)
[x_2, x_partial]=denoi(x_1,sigma_hat);
x_partial = mean(x_partial);
x_3 = (x_2 - x_1.*x_partial)./(1-x_partial);

end


function [x_3,x_2] = DivFree_denoiser4(x_1, r_t, sigma_hat, denoi)
[x_2, x_partial]=denoi(x_1,sigma_hat);
x_3 = (x_2 - r_t.*x_partial);

end


function [x_3,x_2] = DivFree_denoiser5(x_1, z_t, s_t, r_t, y, sigma_hat, A, At, denoi, ori_im, i)
N = length(x_1);
M = length(y);
x_2=denoi(x_1,sigma_hat);
ht = x_1-ori_im(:);
% vht1 = var(ht);
%vht2 = norm(ht,2).^2/N;
% vht3 = norm(y'-A(x_1)).^2/N;
vht4 = (0.5*(norm(z_t,2).^2/N+norm(y'-A(x_1)).^2/N));
Div1 = Div_function1(x_1, denoi, sqrt(vht4));
Div2 = Div_function2(x_2, x_1, s_t, y, A, At, vht4);
Div3 = Div_function3(x_2, z_t, ori_im, ht);
Div4 = Div_function4(x_2, r_t);

if isequal(imag(Div2),0)
    Div = Div2;
else
    %Div2 = mean(real(Div2));%Div_function1(x_1, denoi, sqrt(vht4));
    Div = mean(real(Div2));
    fprintf('Div2 is complex: ');
end
fprintf('Div1=%f, Div2=%f, Div3=%f, Div4=%f\n', Div1, mean(real(Div2)), Div3, Div4);
if mod(i,4)==1 % 0, 3, 5, 7 
    Div = Div_function1(x_1, denoi, sqrt(vht4));
    %x_3 = x_2 - r_t.*Div1; 
    x_3 = (x_2 - x_1.*Div)/(1-Div);     
    fprintf('Using MC \n');
else
    x_3 = x_2 - z_t.*Div;   
end
% Div = Div2;
% x_3 = (x_2 - x_1.*Div)/(1-Div);     

%x_3 = (x_2 - z_t.*Div2);

% ht_qt = (x_1-ori_im(:))'*(x_3-ori_im(:))/N;
% ht_qt_2 = norm(y'-A(x_3),2).^2*(1/M)+vht1-norm(x_3-x_1,2)^2/N;
% %ht_qt_2 = norm(x_3-ori_im(:),2)^2+norm(ht,2).^2-norm(x_3-x_1,2)^2;
% ht_qt_2 = ht_qt_2/2;
% fprintf('qt*ht=%f, ht_qt_2=%f\n', ht_qt, ht_qt_2);

end

function [x_3,x_2] = DivFree_denoiser6(x_1, z_t, s_t, r_t, y, sigma_hat, A, At, denoi, ori_im, i)




end



function Div = Div_function1(x, denoi, sigma_hat)

n = size(x,1);
eta=randn(1,n);
epsilon = 1;
x_1 = denoi(x,sigma_hat);
x_2 = denoi(x+eta'*epsilon,sigma_hat);
Div=1/n*eta*(x_2-x_1)/epsilon;

end

function Div = Div_function2(gt, rt, st, y, A, At, vht)
N = length(gt(:));
M = length(y(:));
st_1 = st{end};
u0_1 = norm(rt-gt,2).^2 - N/M*norm(y-A(gt)',2).^2 - N*vht;
%u1_1 = (rt-st_1)'*(rt-gt)-(rt-st_1)'*At(y'-A(gt));
u1_1 = 2*(rt-st_1)'*(rt-gt)-2*N/M*(y'-A(gt))*A(rt-st_1)';
u2_1 = norm(rt-st_1,2).^2-N/M*norm(A(rt-st_1),2).^2;

%u0_1 = order_compensatory(u2_1, u1_1, u0_1);


a_1 = roots([u2_1, u1_1, u0_1]);

%fprintf('a1=%f, %f, ', a_1(1), a_1(2));
st_2 = st{end-1};
u0_2 = norm(rt-gt,2).^2 - N/M*norm(y-A(gt)',2).^2 - N*vht;
%u1_2 = (rt-st_2)'*(rt-gt)-(rt-st_2)'*At(y'-A(gt));
u1_2 = 2*(rt-st_2)'*(rt-gt)-2*N/M*(y'-A(gt))*A(rt-st_2)';
u2_2 = 1*norm(rt-st_2,2).^2-N/M*norm(A(rt-st_2),2).^2;

%u0_2 = order_compensatory(u2_2, u1_2, u0_2);

a_2 =roots([u2_2, u1_2, u0_2]);
%fprintf('a2=%f, %f\n', a_2(1), a_2(2));

a_1 = repmat(a_1, [1, 2]);
a_2 = repmat(a_2, [1,2]);
res = a_1-a_2';
min_value = min(min(abs(res)));
[row, col] = find(abs(res) == min_value);
a_k = a_1(row);
a_s = a_2(col);
Div = (a_k+a_s)/2;


end




function Div = Div_function3(gt, rt, ori_im, ht)
N = length(gt(:));
Div = ht'*(gt-ori_im(:))/(ht'*rt);

end

function Div = Div_function4(gt, rt_past)
n = length(rt_past);
if n>=3
   rt_bar = rt_past(end-2:end);
else
   rt_bar = rt_past(end-1); 
end
rt_bar = mean(cell2mat(rt_bar),2);
rt = rt_past{end};
Div_up = (rt-rt_bar)'*gt;
Div_down = (rt-rt_bar)'*rt;
Div = Div_up/Div_down;
end






function c = order_compensatory(a, b, c)
x = 0;
if b^2-4*a*c<=0
   x = (b^2+100)/(4*a)-c; 
end
c = x+c;


end


