function [rec_im,PSNR_nlr,SSIM_nlr,PSNR_spl,SSIM_spl]   =  NLR_CS_Reconstruction( y, A, At, measure, par)
ori_im=imread(measure.Test_image_dir);
ori_im=double(ori_im);% For computing PSNR only  
%%

%%
num_cols=measure.image_width;
num_rows=measure.image_height;
Phi=measure.Phi;
num_levels=5;
rec_im0  = BCS_SPL_DDWT_Decoder(y, Phi, num_rows, num_cols,num_levels,200,A,At);
PSNR_spl     =   csnr( rec_im0, ori_im, 8, 0, 0 );
SSIM_spl      =  cal_ssim( rec_im0, ori_im, 0, 0 );
fprintf( 'BCS-SPL : PSNR = %f; SSIM=%f \n',PSNR_spl,SSIM_spl );
%%
rec_im           =    rec_im0;
AtY              =    At(y);
beta             =    par.beta;  %0.01;
[h, w]           =    size( rec_im );
cnt              =    0;
iters            =    15;
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
    
    for it  =  1 : iters
        cnt      =   cnt  +  1;      
        [rim, wei, U_arr]      =   Low_rank_appro(f, par, blk_arr, U_arr, it, flag );
        rim     =    (rim+beta*f)./(wei+beta);
      
%         PSNR     =   csnr( rim, ori_im, 8, 0, 0 );
%         fprintf( '   %d interation_1 : PSNR = %f \n', cnt, PSNR );
       
%%
        b               =   AtY + beta * rim(:);
        [X flag0]       =   pcg( @(x) Afun(x, At, A, beta, wei(:)), b, 0.5E-6, 400, [], [], f(:));
        f               =   reshape(X, h, w);
      
%         PSNR     =   csnr( f, ori_im, 8, 0, 0 );
%         fprintf( '   %d interatio_2 : PSNR = %f \n\n', cnt,PSNR );
       

    end
    rec_im    =   f;

end

rec_im=rim;

PSNR_nlr     =   csnr( rec_im, ori_im, 8, 0, 0 );
SSIM_nlr      =  cal_ssim( rec_im, ori_im, 0, 0 );


return;




%%
function  y  =  Afun(x, At, A, eta, Wei)
%y      =   At( A(x) ) + eta*x;  % eta * (Wei.*x);
y      =   At( A(x) ) + eta *x;
return;
