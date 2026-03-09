function output=SigEstmate_SigCNN(noisy)
global global_time
name='SigCNN.mat';
useGPU = 1;
load(name);%loads net
P_scale = max(max(max(abs(noisy))));


if useGPU == 1
    
    
    noisy = single(noisy./(P_scale./2)-1);
    noisy=real(noisy);
    input = gpuArray(noisy);
    net = vl_simplenn_move(net, 'gpu') ;
    tic
    res = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    output = res(end).x;
    Time_temp=toc;
    global_time = global_time+Time_temp;
    output = double(gather(output*P_scale/2));
    
    
else
    
    noisy = noisy/(255/2)-1;
    input = noisy;
    res = simplenn_matlab(net, input);
    output = res(end).x;
    
end


end