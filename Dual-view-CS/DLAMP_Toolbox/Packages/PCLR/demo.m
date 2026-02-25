%clear all;
addpath(genpath('../PCLR'));
x      =  double(imread('peppers.png'));
sigma  =  30;
randn('seed', 0 );
rand ('seed', 0 );
y      =   x + randn(size(x))*sigma; %Generate noisy image
[z] = PCLR( y,sigma );
PSNR=psnr(z,x);
fprintf( 'Our result. Final PSNR %2.2f\n', PSNR);



