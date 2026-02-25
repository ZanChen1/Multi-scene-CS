% 
% % 生成示例数据
% x = randn(100, 2);  % 输入特征，100个样本，每个样本有2个特征
% y = randi([0, 1], 100, 1);  % 输出标签，二分类问题
% 
% % 创建前馈神经网络
% net = feedforwardnet(10);  % 10个隐藏层神经元
% 
% % 训练神经网络
% net = train(net, x', y');flops
% 
% % 测量神经网络参数
% weights = net.IW;  % 输入层权重矩阵
% biases = net.b;    % 偏置项矩阵
% 
% % 显示权重和偏置项
% disp('输入层权重矩阵:');
% disp(weights{1});
% disp('偏置项矩阵:');
% disp(biases{1});

% 假设已经训练好的神经网络对象为 net
% ...
denoise_name_all = { 'DnCNN_20Layers_3x3_17cases',...
                     'EDSR_5x5_17cases',...
                     'RCAN_5x5_dilated_17cases',...
                     'DPIR',...
                     'MWCNN24',...
                     'Restormer',...
                     'BM3D'};

profile on;
main_parameters
profileStruct = profile('info');
FLOPS('main_parameters','exampleScriptMAT',profileStruct)



% 
% % 获取神经网络的所有层
% layers = net.Layers;
% 
% % 计算权重和偏置的总数量
% totalParams = 0;
% for i = 1:length(layers)
%     if isprop(layers(i), 'Weights') && ~isempty(layers(i).Weights)
%         totalParams = totalParams + numel(layers(i).Weights);
%     end
%     if isprop(layers(i), 'Bias') && ~isempty(layers(i).Bias)
%         totalParams = totalParams + numel(layers(i).Bias);
%     end
% end
% 
% disp(['训练好的神经网络的参数量：', num2str(totalParams)]);
