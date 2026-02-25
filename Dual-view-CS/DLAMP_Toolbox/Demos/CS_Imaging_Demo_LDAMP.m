%Demonstrates compressively sampling and LD(V)AMP recovery of an image.
%Requires  matconvnet and gampmatlab in the path

addpath(genpath('..'));
addpath(genpath('..\gampmatlab'));
addpath('..\..\matconvnet-1.0-beta25/matlab/mex');
addpath('..\..\matconvnet-1.0-beta25/matlab/simplenn');
addpath(genpath('../../Trained_Weights'));
%Parameters
denoiser1='RCAN_5x5_dilated_17cases';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, BM3D-SAPCA, and DnCNN
denoiser2='RCAN_5x5_dilated_17cases';
filename='04.png';
SamplingRate=.1;
AMP_iters=10;
VAMP_iters=10;
imsize=256; 
n_DnCNN_layers=20;%Other option is 17
%LoadNetworkWeights(n_DnCNN_layers);

ImIn=double(imread(filename));
%ImIn = rgb2gray(ImIn);
x_0=imresize(ImIn,imsize/size(ImIn,1));
[height, width]=size(x_0);
n=length(x_0(:));
m=round(n*SamplingRate);
errfxn = @(x_hat) PSNR(x_0,reshape(x_hat,[height width]));


%Generate Coded Diffraction Pattern Measurement Matrix
signvec = exp(1i*2*pi*rand(n,1));
inds=[1;randsample(n-1,m-1)+1];
I=speye(n);
SubsampM=I(inds,:);
M=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*(1/sqrt(n));%*(1/sqrt(n));*(1/sqrt(n))*sqrt(n/m)
Mt=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*sqrt(m);%*sqrt(n);*sqrt(n)*sqrt(n/m)
U=@(x) x(:);
Ut= @(x) x(:);
d=ones(m,1)*n/m;

%%
% SubsampM=randn(m,n);
% for j = 1:n
%     SubsampM(:,j) = SubsampM(:,j) ./ sqrt(sum(abs(SubsampM(:,j)).^2));
% end
% external_SVD = false; 
% 
% if external_SVD
%   [U,D] = eig(SubsampM*SubsampM');
%   Mt = @(z) SubsampM'*z;
%   M = @(x) SubsampM*x;
%   Ut = @(z) U'*z;
%   U = @(x) U*x;
%   d = diag(D);
% else
%   Mt = [];
%   U = [];
%   Ut = [];
%   d = [];
% end
% Mt = @(z) SubsampM'*z;
% M = @(x) SubsampM*x;
%%
%Compressively sample the image

y=M(x_0(:));

%Recover Signal using D-AMP algorithms
t0=tic;[x_hat1,psnr1]  = DAMP(y,AMP_iters,height,width,denoiser1,M,Mt,[],errfxn);t1=toc(t0);
t0=tic;[x_hat2,psnr2] = DVAMP(y,VAMP_iters,height,width,denoiser1,M,Mt,errfxn, U, Ut, d);t2=toc(t0);

%D(V)AMP Recovery Performance
performance1=PSNR(x_0,x_hat1);
performance2=PSNR(x_0,x_hat2);
display([num2str(SamplingRate*100),'% Sampling ', denoiser1, '-AMP: PSNR=',num2str(performance1),', time=',num2str(t1)])
display([num2str(SamplingRate*100),'% Sampling ', denoiser2, '-VAMP: PSNR=',num2str(performance2),', time=',num2str(t2)])


%Plot Recovered Signals
figure(1); clf;
subplot(1,3,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,3,2);
imshow(uint8(x_hat1));title([denoiser1, '-AMP']);
subplot(1,3,3);
imshow(uint8(x_hat2));title([denoiser2, '-VAMP']);

%Plot PSNR Trajectories
figure(2); clf;
plot(psnr1,'.-','Displayname',[denoiser1,'-AMP'])
hold on; 
plot(psnr2,'.-','Displayname',[denoiser2,'-VAMP']); 
hold off;
grid on;
legend(gca,'Show','Location','SouthEast')
xlabel('iteration')
ylabel('PSNR')