function [Re]=Dec_main()
%close all;

addpath('..\Utilities');
addpath('..\Utilities\Measurements');
addpath('..\Quantize')
addpath(genpath('..\BCS\BCS-SPL-1.5-1'));
addpath(genpath('..\BCS\BCS-SPL-DPCM-1.0-2'));
addpath('..\BCS\WaveletSoftware');
%%
load('..\channel\transmit_data.mat');
%bpp=(length(cell2mat(Trans.data(1)))*quantize.bit(1)+length(cell2mat(Trans.data(2)))*quantize.bit(2)+length(Trans.data{3})*quantize.bit(3))/measure.image_width/measure.image_height;
quantize.Rate_proportion;
%fprintf('recieve bpp=%d\n',bpp);
%%  
[Re]=CS_decode(Trans,quantize,measure);
%Re.PLR=quantize.PLR;



end

