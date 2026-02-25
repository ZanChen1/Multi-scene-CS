function []=Enc_main(Image_name,bpp,quantize)

delete('..\channel\transmit_data.mat')
%%
addpath('..\Utilities');
addpath('..\Utilities\Measurements');
addpath('..\Quantize')
addpath(genpath('..\BCS\BCS-SPL-1.5-1'));
addpath(genpath('..\BCS\BCS-SPL-DPCM-1.0-2'));
addpath('..\BCS\WaveletSoftware');
addpath('..\entropy_code');
%% load original images
Test_image_dir = 'D:\cz\Test_Images\256\1\';
ori_im = imread( fullfile(Test_image_dir, Image_name)) ;
ori_im=double(ori_im);


%% Quantization parameter setting
quantize.Mu=[];
quantize.Sigm=[];
quantize.distr='Gaussian';%'Gaussian'or'Uniform','Student','Train'
quantize.method='BAQ';%'BAQ'
% quantize.packsize=[90/quantize.bit(1),90/quantize.bit(2),90/quantize.bit(3),90/quantize.bit(4)];

%% measurement parameter setting
measure.Test_image_dir = fullfile(Test_image_dir, Image_name);
measure.Image_name=Image_name;
measure.model='Gaussian';%'Hadamard_matrix'or'Orthogonal','Dct_matrix','FFT','Gaussian'


[measure.image_width,measure.image_height]=size(ori_im);
Level=quantize.Level;
for i=1:Level
    measure.rate_allocation(i)=round(measure.image_width*measure.image_height*bpp*quantize.Rate_proportion(i)/quantize.bit(i));
end
[measure.prograssive_step]=OMEGA_slip(measure.rate_allocation);
q=1:length(ori_im(:));
step=measure.prograssive_step;
measure.OMEGA=cell([1,length(step)]);
for i=1:length(measure.prograssive_step)
    measure.OMEGA{i}= q(step(i,1):step(i,2));
end
%% Gaussian measurement matrix
m = sum(measure.rate_allocation);
n = numel(ori_im);
% Gaussian_measure_dir = '../Gaussian_measure/';
% if ~exist(Gaussian_measure_dir,'dir')
%     mkdir(Gaussian_measure_dir);
% end
% filename = [Gaussian_measure_dir,num2str(m),'_',num2str(n),'.mat'];
% if exist(filename,'file')
%     load(filename);
% else
%     randn('state',1);
%     rand('state',1);
%     SubsampM=randn(m,n);
%     for j = 1:n
%         SubsampM(:,j) = SubsampM(:,j) ./ sqrt(sum(abs(SubsampM(:,j)).^2));
%     end
%     save(filename, 'SubsampM', '-v7.3');
% end
% Measure.SubsampM = SubsampM;
% At = @(x) SubsampM(cell2mat(measure.OMEGA),:)'*x(:);
% A = @(x) SubsampM(cell2mat(measure.OMEGA),:)*x(:);
%% diffraction measurement matrix
rand('state',1)
randn('state',1)
width = measure.image_width;
height = measure.image_height;
signvec = exp(1i*2*pi*rand(n,1));
inds=[1;randsample(n-1,m-1)+1];
I=speye(n);
SubsampM=I(inds,:);
A=@(x) real(SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*sqrt(1/m));
At=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*n*sqrt(1/m);
U=@(x) x(:);
Ut= @(x) x(:);
d=ones(m,1)*n/m;

%% ≤‚¡ø
par.y = A(ori_im);
[par,quantize]=quantize_cell(par,quantize,measure.OMEGA,1);
[par,quantize]=quantize_cell(par,quantize,measure.OMEGA,0);
bin=par.bin(1);
Mu=quantize.Mu(1);
Sigm=quantize.Sigm(1);
huff_enc = mat2huff(cell2mat(bin(1)));
Trans.data{1}=huff_enc.code;
Trans.hist{1}=huff_enc.hist;
quantize.size{1}=huff_enc.size;

%% ‘§≤‚
AMP_iters=10;
y=cell2mat(par.dec(1));
for L=1:Level-1
    
    p_t = sqrt(m/length(y));
    OMEGA=cell2mat(measure.OMEGA(1:L));
    %     At = @(x) SubsampM(OMEGA,:)'*x(:);
    %     A = @(x) SubsampM(OMEGA,:)*x(:);
    A=@(x) real(SubsampM(OMEGA,:)*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*sqrt(1/length(OMEGA)));
    At=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM(OMEGA,:)'*x(:),[height,width])),[n,1]))*n*sqrt(1/length(OMEGA));
    %[par]=CS_predict(y, OMEGA, par, measure, quantize.distr, quantize.bit(1:L+1));
    
    [par.predict_im,~]  = DAMP(y',AMP_iters,measure.image_height,measure.image_width,'DnCNN',A,At);
    fprintf('L = %d, Psnr = %f\n',L,csnr(par.predict_im, ori_im, 0, 0));
    predict_im{L}=par.predict_im;
    
    
    %% ≤–≤Ó
    
    OMEGA=cell2mat(measure.OMEGA(L+1));
    %     At = @(x) SubsampM(OMEGA,:)'*x(:);
    %     A = @(x) SubsampM(OMEGA,:)*x(:);
    A=@(x) real(SubsampM(OMEGA,:)*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*sqrt(1/m));
    par.y{L+1}=A(ori_im)-A(predict_im{L});
    
    %% ¡øªØ
    
    
    [par,quantize]=quantize_cell(par,quantize,measure.OMEGA,1);
    [par,quantize]=quantize_cell(par,quantize,measure.OMEGA,0);
    bin(L+1)=par.bin(L+1);
    Mu(L+1)=quantize.Mu(L+1);
    Sigm(L+1)=quantize.Sigm(L+1);
    
    
    huff_enc = mat2huff(cell2mat(bin(L+1)));
    Trans.data{L+1}=huff_enc.code;
    Trans.hist{L+1}=huff_enc.hist;
    quantize.size{L+1}=huff_enc.size;
    quantize.min=huff_enc.min;
    
    y_t = double(A(predict_im{L}));
    y=[y,y_t'+par.dec{L+1}];
    
end
L=L+1;
OMEGA=cell2mat(measure.OMEGA(1:L));
% At = @(x) SubsampM(OMEGA,:)'*x(:);
% A = @(x) SubsampM(OMEGA,:)*x(:);
A=@(x) real(SubsampM(OMEGA,:)*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*(1/sqrt(n))*sqrt(n/m));
At=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM(OMEGA,:)'*x(:),[height,width])),[n,1]))*sqrt(n)*sqrt(n/m);
[par.predict_im,~]  = DAMP(y',AMP_iters,measure.image_height,measure.image_width,'DnCNN',A,At);
fprintf('L = %d, Psnr = %f\n',L,csnr(par.predict_im, ori_im, 0, 0));
predict_im{L}=par.predict_im;

Trans.Mu=Mu;
Trans.Sigm=Sigm;
quantize=rmfield(quantize,'Mu');
quantize=rmfield(quantize,'Sigm');

save('..\channel\transmit_data.mat','Trans','quantize','measure')





end