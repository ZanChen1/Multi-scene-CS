function [output] = denoise_TNRD(noisy, sigma_hat)

if sigma_hat == 5
    name = 'JointTraining_7x7_400_180x180_stage=5_sigma=5.mat';
elseif sigma_hat == 15
    name = 'JointTraining_7x7_400_180x180_stage=5_sigma=15.mat';
elseif sigma_hat == 25
    name = 'JointTraining_7x7_400_180x180_stage=5_sigma=25.mat';
elseif sigma_hat == 50
    name = 'JointTraining_7x7_400_180x180_stage=5_sigma=50.mat';
else
    name = 'JointTraining_7x7_400_180x180_stage=5_sigma=50.mat';
end
load(name);
%%
filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;
BASIS = gen_dct2(filter_size);
BASIS = BASIS(:,2:end);
%% pad and crop operation
bsz = 8;
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
%% MFs means and precisions
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% MFs means and precisions
trained_model = TNRD_save_trained_model(cof, MFS, stage, KernelPara);
Im = noisy;
%% run denoising, 5 stages
input = pad(Im);
noisy = pad(Im);
run_stage = 5;

for s = 1:run_stage
    deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
    t = crop(deImg);
    deImg = pad(t);
    input = deImg;
end
output = t;


end