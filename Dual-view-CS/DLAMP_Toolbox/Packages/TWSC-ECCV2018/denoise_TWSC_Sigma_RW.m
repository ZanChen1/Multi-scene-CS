%-------------------------------------------------------------------------------------------------------------
% This is an implementation of the TWSC algorithm for real-world image denoising
%
% Author:  Jun Xu, csjunxu@comp.polyu.edu.hk / nankaimathxujun@gmail.com
%          The Hong Kong Polytechnic University
%
% Please refer to the following paper if you find this code helps:
%
% @article{TWSC_ECCV2018,
% 	author = {Jun Xu and Lei Zhang and David Zhang},
% 	title = {A Trilateral Weighted Sparse Coding Scheme for Real-World Image Denoising},
% 	journal = {ECCV},
% 	year = {2018}
% }
%
% Please see the file License.txt for the license governing this code.
%-------------------------------------------------------------------------------------------------------------
function  [IMout] = denoise_TWSC_Sigma_RW(noise_img)

method = 'TWSC';

% Parameters
Par.ps = 6;        % patch size
Par.step = 3;       % the step of two neighbor patches
Par.win = 20;   % size of window around the patch
Par.Outerloop = 8;
Par.Innerloop = 2;
Par.nlspini = 70;
Par.display = 0;
Par.delta = 0;
Par.nlspgap = 10;
Par.lambda1 = 0;
Par.lambda2 = 1; % set randomly as 1, different for each image
% set Parameters


Par.nlsp = Par.nlspini;  % number of non-local patches

Par.nim = noise_img;
Par.I = noise_img; % just for consistency in 'TWSC_Sigma_RW.m' function

[~,~,ch] = size(Par.nim);
% noise estimation
for c = 1:ch
    Par.nSig(c) = NoiseEstimation(Par.nim(:, :, c)*255, Par.ps)/255;
end

% denoising
t1=clock;
[IMout, ~]  =  TWSC_Sigma_RW(Par);
t2=clock;

% alltime  = etime(t2, t1);
% fprintf('TWSC Iterate once %s:\n', alltime );
%% output

