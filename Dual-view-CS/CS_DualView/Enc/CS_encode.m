function  [par,measure]=CS_encode(ori_im,measure)
randn('state',0);
rand('state',0);
%%
[measure]=Compressive_sensing(measure);
A_Y=@(z)A_bp2(z,measure.OMEGA,measure.P_image,measure.P_block,measure.Phi);
par.y=A_Y(ori_im(:));

return;



function [measure]=Compressive_sensing(measure)

    rand('seed',0);   
    h=measure.image_height;
    w=measure.image_width;
    N=h*w;
    P_image=randperm(N);
    block_size=measure.block_size;
    P_block=randperm(block_size*block_size);
    Phi=NLR_GenerateProjection(measure.block_size,measure.model);
    measure.P_image=P_image;
    measure.P_block=P_block;
    measure.Phi=Phi;
      
return;
