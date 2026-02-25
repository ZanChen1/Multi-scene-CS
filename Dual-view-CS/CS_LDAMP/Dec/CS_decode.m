function  [Re]    =  CS_decode(Trans,quantize,measure)

ori_im=imread( measure.Test_image_dir);
ori_im=double(ori_im);% For computing PSNR only
%%
randn('state',0);
rand('state',0);
AMP_iters = 10;
denoize_name = 'DnCNN17';
Phi = measure.Phi;
errfxn = @(x_hat) PSNR(ori_im,reshape(x_hat,[measure.image_height,measure.image_width]));
% bin=Trans.data;
% huff_enc.min=quantize.min;
% huff_enc.code=bin{1};
% huff_enc.hist=Trans.hist{1};
% huff_enc.size=quantize.size{1};
% data_temp=huff2mat(huff_enc);
% Trans.data{1}=data_temp(measure.OMEGA{1});
% Trans.data{2}=data_temp(measure.OMEGA{2});
% for L=2:Level
%     
%     huff_enc.code=bin{L};
%     huff_enc.hist=Trans.hist{L};
%     huff_enc.size=quantize.size{L};
%     Trans.data{L+1}=huff2mat(huff_enc);
% end
par.bin=Trans.bin(1);
quantize.Mu=Trans.Mu;
quantize.Sigm=Trans.Sigm;
Level=quantize.Level;
%% 测试
[par,quantize]=quantize_cell(par,quantize,measure.OMEGA,0);
par.y{1} = par.dec{1};
y = par.dec{1};
for L=1:Level-1
    
    OMEGA=cell2mat(measure.OMEGA(1:L));
    %%
    k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
    Phi_mt = zeros(k,measure.block_size^2);
    for j = 1:measure.block_size^2
        Phi_mt(1:k,j) = Phi(1:k,j)./(sum(abs(Phi(1:k,j)).^2));
    end
    measure.Phi_mt = Phi_mt;
    Phi_mp = pinv(Phi(1:k,:));
    measure.Phi_mp = Phi_mp';
    A=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
    At=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi_mt);
    
    %%
    
    [measure.predict_im,~]  = DAMP(y',AMP_iters,measure.image_height,measure.image_width,denoize_name,A,At,measure, OMEGA,errfxn);
    fprintf('L = %d, Psnr = %f\n',L,csnr(measure.predict_im, ori_im,8, 0, 0));
    OMEGA=cell2mat(measure.OMEGA(L+1));
    A=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
    par.y{L+1}=A(measure.predict_im);
    
    %% 量化    
    [par,quantize]=quantize_cell(par,quantize,measure.OMEGA,1);
    par.bin{L+1} =Trans.bin{L+1}+double(par.bin{L+1});
    quantize.Mu=Trans.Mu;
    quantize.Sigm=Trans.Sigm;
    [par,quantize]=quantize_cell(par,quantize,measure.OMEGA,0);
    
    %% 量化
    
    y=[y,par.dec{L+1}];
    
    
end
L=L+1;
OMEGA=cell2mat(measure.OMEGA(1:L));
A=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
k = ceil(length(OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
Phi_mt = zeros(k,measure.block_size^2);
for j = 1:measure.block_size^2
    Phi_mt(1:k,j) = Phi(1:k,j)./(sum(abs(Phi(1:k,j)).^2));
end
measure.Phi_mt = Phi_mt;
Phi_mp = pinv(Phi(1:k,:));
measure.Phi_mp = Phi_mp';
At=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi_mt);
%%
[measure.predict_im,~]  = DAMP(y',AMP_iters,measure.image_height,measure.image_width,denoize_name,A,At,measure,OMEGA,errfxn);
%[par]=CS_predict(y, OMEGA, par, measure, quantize.distr, quantize.bit(1:L));
fprintf('L = %d, Psnr = %f\n',L,csnr(measure.predict_im, ori_im, 8,0, 0));
predict_im{L}=measure.predict_im;
%%
rec_im=predict_im{L};
Re.Rec_im=rec_im;



return;




