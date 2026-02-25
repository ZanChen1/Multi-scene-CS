



k_t = 0;
layer_record = struct();
for i = 1:length(net.layers)
    tmp = net.layers{i};
    switch tmp.type
        case{'conv','convt','bnorm','pool','normalize','lrn','relu','sigmoid','dropout'}
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
        case{'conv1x1'}
            net.layers{i}.type = 'conv';
            [b,c] = size(tmp.weights{1});
            net.layers{i}.weights{1} = reshape(tmp.weights{1},[1,1,b,c]);
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
        case{'addrca'}
            net.layers{i}.type = 'cross_layer_add';
            net.layers{i}.stride = 9;
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
        case{'addgroup'}
            net.layers{i}.type = 'cross_layer_add';
            net.layers{i}.stride = 81;
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
        case{'addall'}
            net.layers{i}.type = 'cross_layer_add';
            net.layers{i}.stride = 165;
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
        case{'multi'}
            net.layers{i}.type = 'cross_layer_multiplication';
            net.layers{i}.stride = 5;
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
        case{'reduce_mean'}
            net.layers{i}.type = 'mean_filling';
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
            
        otherwise
            k_t = k_t+1;
            layer_record.location(k_t) = i;
            layer_record.type{k_t} = tmp.type;
            
            
    end
end

