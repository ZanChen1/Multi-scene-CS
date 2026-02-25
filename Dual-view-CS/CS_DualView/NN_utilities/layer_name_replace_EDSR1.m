
fileFolder_input = fullfile('../../Trained_Weights/EDSR/edsr_3x3_10/backup/');
fileFolder_output = fullfile('../../Trained_Weights/EDSR/edsr_3x3_10/');

dir_name = dir(fullfile(fileFolder_input,'*.mat'));
dir_number = length(dir_name);
for j = 1:dir_number
    load(fullfile(fileFolder_input,dir_name(j).name));
    
    
    k_t = 1;
    layer_record = struct();
    
    for i = 1:length(net.layers)
        
        
        
        layer_record.layers{k_t} = net.layers{i};
        layer_record.layers{k_t}.name = ['layer',int2str(k_t)];
        k_t = k_t+1;
        if i-1>=1
            
            if strcmp(net.layers{i}.type,'conv') && strcmp(net.layers{i-1}.type,'relu')
                layer_record.layers{k_t}.type = 'cross_layer_add';
                layer_record.layers{k_t}.stride = 3;
                layer_record.layers{k_t}.precious = 0;
                layer_record.layers{k_t}.name = ['layer',int2str(k_t)];
                k_t = k_t+1;
                
            end
            
        else
            
        end
        
    end
    layer_record.layers{k_t}.type = 'cross_layer_add';
    layer_record.layers{k_t}.stride = 38;
    net = layer_record;
    if k_t~=0
        save(fullfile(fileFolder_output, dir_name(j).name),'net');
    end
    
end


