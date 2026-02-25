randn('state',0);
rand('state',0);



measure.ori_im = im2double(imread('D:\Chenzan\CS_image\Test_Images\Set\Set8\1.tif'));
ori_im = measure.ori_im;
measure.rate = 0.1;

[measure.image_width , measure.image_height]=size(measure.ori_im);
measure.block_size = 64;
measure.P_image=randperm(measure.image_height*measure.image_width);
measure.P_block=randperm(measure.block_size*measure.block_size);
measure.rate_allocation = ceil(measure.image_width*measure.image_height*measure.rate);
q=1:(measure.image_width*measure.image_height);
step(1,1) = 1;
step(1,2) = measure.rate_allocation;
measure.OMEGA = q(step(1,1):step(1,2));


%%
k = ceil(length(measure.OMEGA)/(measure.image_height*measure.image_width/measure.block_size^2));
p = 0.5;
Phi_B = double((rand(measure.block_size^2, measure.block_size^2)<p));
Phi_B = Phi_B*2-1;
Phi_B = Phi_B./(measure.block_size);
Phi=Phi_B;
Phi_mt = Phi_B*(measure.block_size^2/k);
measure.Phi = Phi;
measure.Phi_mt = Phi_mt;
%%

A_1=@(z)A_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi);
At_1=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mt);



ori_im_keys = ori_im(1:128,1:128);
[measure.image_width, measure.image_height]=size(ori_im_keys);
measure.P_image=randperm(measure.image_height*measure.image_width);
measure.P_block=randperm(measure.block_size*measure.block_size);

%%

A_2=@(z)A_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi);
At_2=@(z)At_bp(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi_mt);

y_1 = A_1(ori_im);
y_2 = A_2(ori_im_keys);
y = 0.5*(y_1+y_2);




