function [] = LoadNetworkWeights(Net_Name,n_layers)
    %Load the shared NN-based denoisers network
    if ~exist('vl_setupnn','file')
        error('vl_setupnn not found. Make sure matconvnet is in your path');
    end
    vl_setupnn
    global net_0to5
    global net_5to10
    global net_10to15
    global net_15to20
    global net_20to30
    global net_30to40
    global net_40to50
    global net_50to60
    global net_60to70
    global net_70to80
    global net_80to90
    global net_90to100
    global net_100to125
    global net_125to150
    global net_0to10
    global net_10to20
    global net_20to40
    global net_40to60
    global net_60to80
    global net_80to100
    global net_100to150
    global net_150to300
    global net_300to500
    global net_500to1000
    global net1_0to5
    global net1_5to10
    global net1_10to15
    global net1_15to20
    global net1_20to30
    global net1_30to40
    global net1_40to50
    global net1_50to60
    global net1_60to70
    global net1_70to80
    global net1_80to90
    global net1_90to100
    global net1_100to125
    global net1_125to150
    global net1_150to300
    global net1_300to500
    global net1_500to1000
    global net2_0to5
    global net2_5to10
    global net2_10to15
    global net2_15to20
    global net2_20to30
    global net2_30to40
    global net2_40to50
    global net2_50to60
    global net2_60to70
    global net2_70to80
    global net2_80to90
    global net2_90to100
    global net2_100to125
    global net2_125to150
    global net2_150to300
    global net2_300to500
    global net2_500to1000    
    
        
    switch Net_Name
        
        case 'DnCNN17'
            
            
        case 'EDSR5x5'
            
        case 'RCAN'
            
    end
    
    

    if nargin~=0
        if n_layers==17
            %Load 17 layer network weights
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma0to10-best');%loads net
            net.layers = net.layers(1:end-1);
            net_0to10 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma10to20-best');%loads net
            net.layers = net.layers(1:end-1);
            net_10to20 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma20to40-best');%loads net
            net.layers = net.layers(1:end-1);
            net_20to40 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma40to60-best');%loads net
            net.layers = net.layers(1:end-1);
            net_40to60 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma60to80-best');%loads net
            net.layers = net.layers(1:end-1);
            net_60to80 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma80to100-best');%loads net
            net.layers = net.layers(1:end-1);
            net_80to100 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma100to150-best');%loads net
            net.layers = net.layers(1:end-1);
            net_100to150 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma150to300-best');%loads net
            net.layers = net.layers(1:end-1);
            net_150to300 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma300to500-best');%loads net
            net.layers = net.layers(1:end-1);
            net_300to500 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma500to1000-best');%loads net
            net.layers = net.layers(1:end-1);
            net_500to1000 = vl_simplenn_move(net, 'gpu') ;
        else
            LoadNetworkWeights();
        end
    else
        %Load 20 layer network weights
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma0to10-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_0to10 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma10to20-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_10to20 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma20to40-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_20to40 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma40to60-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_40to60 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma60to80-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_60to80 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma80to100-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_80to100 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma100to150-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_100to150 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma150to300-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_150to300 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma300to500-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_300to500 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma500to1000-best');%loads net
%         net.layers = net.layers(1:end-1);
%         net_500to1000 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_1');%loads net
%         net.layers = net.layers(1:end);
%         net_0to10 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_2');%loads net
%         net.layers = net.layers(1:end);
%         net_10to20 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_3');%loads net
%         net.layers = net.layers(1:end);
%         net_20to40 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_4');%loads net
%         net.layers = net.layers(1:end);
%         net_40to60 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_5');%loads net
%         net.layers = net.layers(1:end);
%         net_60to80 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_6');%loads net
%         net.layers = net.layers(1:end);
%         net_80to100 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_7');%loads net
%         net.layers = net.layers(1:end);
%         net_100to150 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_8');%loads net
%         net.layers = net.layers(1:end);
%         net_150to300 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_9');%loads net
%         net.layers = net.layers(1:end);
%         net_300to500 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_10');%loads net
%         net.layers = net.layers(1:end);
%         net_500to1000 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_1');%loads net
%         net.layers = net.layers(1:end);
%         net_0to10 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_2');%loads net
%         net.layers = net.layers(1:end);
%         net_10to20 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_3');%loads net
%         net.layers = net.layers(1:end);
%         net_20to40 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_4');%loads net
%         net.layers = net.layers(1:end);
%         net_40to60 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_5');%loads net
%         net.layers = net.layers(1:end);
%         net_60to80 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_6');%loads net
%         net.layers = net.layers(1:end);
%         net_80to100 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_7');%loads net
%         net.layers = net.layers(1:end);
%         net_100to150 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_8');%loads net
%         net.layers = net.layers(1:end);
%         net_150to300 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_9');%loads net
%         net.layers = net.layers(1:end);
%         net_300to500 = vl_simplenn_move(net, 'gpu') ;
%         load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_256_10');%loads net
%         net.layers = net.layers(1:end);
%         net_500to1000 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_1');%loads net
        net.layers = net.layers(1:end);
        net_0to5 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_2');%loads net
        net.layers = net.layers(1:end);
        net_5to10 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_3');%loads net
        net.layers = net.layers(1:end);
        net_10to15 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_4');%loads net
        net.layers = net.layers(1:end);
        net_15to20 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_5');%loads net
        net.layers = net.layers(1:end);
        net_20to30 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_6');%loads net
        net.layers = net.layers(1:end);
        net_30to40 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_7');%loads net
        net.layers = net.layers(1:end);
        net_40to50 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_8');%loads net
        net.layers = net.layers(1:end);
        net_50to60 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_9');%loads net
        net.layers = net.layers(1:end);
        net_60to70 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_10');%loads net
        net.layers = net.layers(1:end);
        net_70to80 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_11');%loads net
        net.layers = net.layers(1:end);
        net_80to90 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_12');%loads net
        net.layers = net.layers(1:end);
        net_90to100 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_13');%loads net
        net.layers = net.layers(1:end);
        net_100to125 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_14');%loads net
        net.layers = net.layers(1:end);
        net_125to150 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_15');%loads net
        net.layers = net.layers(1:end);
        net_150to300 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_16');%loads net
        net.layers = net.layers(1:end);
        net_300to500 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model_17_17');%loads net
        net.layers = net.layers(1:end);
        net_500to1000 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_1');%loads net
        net.layers = net.layers(1:end);
        net1_0to5 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_2');%loads net
        net.layers = net.layers(1:end);
        net1_5to10 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_3');%loads net
        net.layers = net.layers(1:end);
        net1_10to15 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_4');%loads net
        net.layers = net.layers(1:end);
        net1_15to20 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_5');%loads net
        net.layers = net.layers(1:end);
        net1_20to30 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_6');%loads net
        net.layers = net.layers(1:end);
        net1_30to40 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_7');%loads net
        net.layers = net.layers(1:end);
        net1_40to50 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_8');%loads net
        net.layers = net.layers(1:end);
        net1_50to60 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_9');%loads net
        net.layers = net.layers(1:end);
        net1_60to70 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_10');%loads net
        net.layers = net.layers(1:end);
        net1_70to80 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_11');%loads net
        net.layers = net.layers(1:end);
        net1_80to90 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_12');%loads net
        net.layers = net.layers(1:end);
        net1_90to100 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_13');%loads net
        net.layers = net.layers(1:end);
        net1_100to125 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_14');%loads net
        net.layers = net.layers(1:end);
        net1_125to150 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_15');%loads net
        net.layers = net.layers(1:end);
        net1_150to300 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_16');%loads net
        net.layers = net.layers(1:end);
        net1_300to500 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/edsr_model1_17_17');%loads net
        net.layers = net.layers(1:end);
        net1_500to1000 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_1');%loads net
        net.layers = net.layers(1:end);
        net2_0to5 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_2');%loads net
        net.layers = net.layers(1:end);
        net2_5to10 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_3');%loads net
        net.layers = net.layers(1:end);
        net2_10to15 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_4');%loads net
        net.layers = net.layers(1:end);
        net2_15to20 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_5');%loads net
        net.layers = net.layers(1:end);
        net2_20to30 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_6');%loads net
        net.layers = net.layers(1:end);
        net2_30to40 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_7');%loads net
        net.layers = net.layers(1:end);
        net2_40to50 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_8');%loads net
        net.layers = net.layers(1:end);
        net2_50to60 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_9');%loads net
        net.layers = net.layers(1:end);
        net2_60to70 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_10');%loads net
        net.layers = net.layers(1:end);
        net2_70to80 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_11');%loads net
        net.layers = net.layers(1:end);
        net2_80to90 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_12');%loads net
        net.layers = net.layers(1:end);
        net2_90to100 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_13');%loads net
        net.layers = net.layers(1:end);
        net2_100to125 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_14');%loads net
        net.layers = net.layers(1:end);
        net2_125to150 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_15');%loads net
        net.layers = net.layers(1:end);
        net2_150to300 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_16');%loads net
        net.layers = net.layers(1:end);
        net2_300to500 = vl_simplenn_move(net, 'gpu') ;
        load('E:\临时文件\AMP\DLAMP_Toolbox/Packages/DnCNN/BestNets_20/rcan_model_17_17');%loads net
        net.layers = net.layers(1:end);
        net2_500to1000 = vl_simplenn_move(net, 'gpu') ;
    end
end

