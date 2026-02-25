function [rec_im]   =  NLR_CS_Reconstruction_predict( y, OMEGA, par, measure)
ori_im=imread( measure.Test_image_dir);
ori_im=double(ori_im);% For computing PSNR only
if iscell(y)
    y=cell2mat(y);
end
if iscell(OMEGA)
    OMEGA=cell2mat(OMEGA);
end
A=@(z)A_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi);
At=@(z)At_bp(z,OMEGA,measure.P_image,measure.P_block,measure.Phi_mp);
%%
num_rows=measure.image_height;
num_cols=measure.image_width;
Phi=measure.Phi;
num_levels=5;

if ~isfield(par,'predict_im')
    rec_im  = BCS_SPL_DDWT_Decoder(y, Phi, num_rows, num_cols, num_levels, 200, A, At);
else 
    rec_im=par.predict_im;
end
%%
%rec_im  = BCS_SPL_DDWT_Decoder(y, Phi, num_rows, num_cols, num_levels, 200, A, At);
AtY              =    At(y);
beta             =    par.beta;  %0.01;
[h, w]           =    size( rec_im );
cnt              =    0;
iters            =    15;%15
%%
for  k    =   1 : par.K
 
    blk_arr      =     Block_matching( rec_im, par);
    f            =     rec_im;
    U_arr        =     zeros(par.win^4, size(blk_arr,2), 'single');
    if (k<=par.K0)  
        flag=0;  
    else
        flag=1;
    end
%%    
    for it  =  1 : iters
        cnt      =   cnt  +  1;      
        [rim, wei, U_arr]      =   Low_rank_appro(f, par, blk_arr, U_arr, it, flag );
        rim     =    (rim+beta*f)./(wei+beta);
        
        PSNR     =   csnr( rim, ori_im, 0, 0 );
        fprintf( '   %d次逼近后 : PSNR = %f, norm = %f \n', cnt, PSNR, norm(rim(:),2) );
       
%%
        b               =   AtY + beta * rim(:);
        [X flag0]       =   pcg( @(x) Afun(x, At, A, beta, wei(:)), b, 0.5E-6, 400, [], [], f(:));
        f               =   reshape(X, h, w);
        
        PSNR     =   csnr( f, ori_im, 0, 0 );
        fprintf( '   %d次引导后 : PSNR = %f norm = %f \n\n', cnt,PSNR,norm(f(:),2) );
        

    end
    %fprintf( '   %d次引导后 : PSNR = %f, error_mean = %f; \n\n', cnt,csnr( f, ori_im, 0, 0 ),mean(mean(abs(rec_im-f))) );
    if mean(mean(abs(rec_im-f)))>=0.01
        rec_im = f;
    else
        break;
    end
end

rec_im = rim;

return;




%%
function  y  =  Afun(x, At, A, eta, Wei)
%y      =   At( A(x) ) + eta*x;  % eta * (Wei.*x);
y      =   At( A(x) ) + eta *x;
return;
