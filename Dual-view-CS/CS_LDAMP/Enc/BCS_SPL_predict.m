function [rec_im]   =  BCS_SPL_predict( y, OMEGA, measure)

if iscell(y)
    y=cell2mat(y);
end
if iscell(OMEGA)
    OMEGA=cell2mat(OMEGA);
end
A=@(z)A_bp2(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
At=@(z)At_bp2(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
%%
num_rows=measure.image_height;
num_cols=measure.image_width;
Phi=measure.Phi;
num_levels=5;
rec_im  = BCS_SPL_DDWT_Decoder(y, Phi, num_rows, num_cols, num_levels, 200, A, At);




return;



