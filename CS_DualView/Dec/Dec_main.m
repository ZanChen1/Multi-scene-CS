function [Re]=Dec_main()
addpath('..\Utilities');
addpath('..\Utilities\Measurements');
addpath('..\Quantize')
addpath(genpath('..\BCS\BCS-SPL-1.5-1'));
addpath(genpath('..\BCS\BCS-SPL-DPCM-1.0-2'));
addpath('..\BCS\WaveletSoftware');
%%
load('..\channel\transmit_data.mat');

quantize.Rate_proportion;

%%  
[Re]=CS_decode(Trans,quantize,measure);

end

