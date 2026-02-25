% This demo implements the stategy for Block based Image Compressive
% Sensing of Paper "Improved Total Variation based Image Compressive
% Sensing Recovery by Nonlocal Regularization".

% x: Image Matrix
% A: Random matrix without normality and orthogonality (real)
% b: Observed measurements with/without noise (real)
%
% Written by: Jian Zhang
% Harbin Institute of Technology
% Email: jzhangcs@hit.edu.cn
% 10/5/2013


clear;
close all;
cur = cd;
addpath(genpath(cur));

for k = 2
    
    Image_name = [num2str(k),'.tif'];
    Test_image_dir = '../../Test_Images/set8/set8/';
    OrgName = fullfile(Test_image_dir, Image_name) ;
    
    
    % Three are two free parameters: theta, beta
    for ratio = 0.1
        for theta = 2 % Set theta from {2,3}             	        ---- parameter 1
            for beta = 5 % Set beta from {5,6,7}            	    ---- parameter 2
                for dd =1
                    x = double(imread(OrgName));
                    [opts] = opts_set(x,ratio,theta,beta);
                    if dd==1
                        x_pre = TVNLR_preprocess(x,opts);
                    else
                        x_pre = x;
                    end
                    csnr(x_pre,x,0,0)
                    % Sensing Matrix
                    A =opts.Phi;
                    
                    % Observed Measurements
                    b = f_handleA(A,x_pre,1,opts);
                    
                    %%
                    
                    [x_Rec, out] = TVNLR(A,b,opts);
                    csnr(x_Rec,x,0,0)
                    
                end
            end
        end
    end
end

