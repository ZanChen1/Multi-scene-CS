

%load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/RCAN_3x3_dilated_17cases/RCAN_3x3_dilated_17cases_sigma500to1000.mat');
%load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/RCAN_5x5_dilated_17cases/rcan_5x5_dilated_17cases_sigma5to10.mat');
% load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/RCAN_5x5_17cases/RCAN_5x5_17cases_sigma500to1000.mat');
% load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/DnCNN/BestNets_17/DnCNN_17Layers_10cases_sigma0to10-best.mat')
% load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/DnCNN/BestNets_20/DnCNN_20Layers_10cases_sigma0to10-best.mat')
%load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/DnCNN/DnCNN_3x3_17/DnCNN_20Layers_3x3_17cases_sigma5to10.mat')
%load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/EDSR/edsr_3x3_10/edsr_3x3_10cases_sigma300to500.mat')
load('C:/Users/zanchen/Desktop/ht/Dual-view-CS/Trained_Weights/EDSR/edsr_5x5_17/edsr_5x5_17cases_sigma300to500.mat')

tmp = 0;

for i=1:length(net.layers)
    
    if isfield(net.layers{i}, 'weights')
        for j=1:length(net.layers{i}.weights)
            
            if sum(net.layers{i}.weights{j}(:)) == 0
                tmp = tmp;
            else
                tmp = tmp+length(net.layers{i}.weights{j}(:));
            end
            
        end
    end
    
    
    
    
end


fprintf('weights number: %d', tmp);