function [par model] = PCLR_ParSet(sigma,x)
    [h  w]        =   size(x);
    sizex         =   0.5*(h+w);
    par.xf        =   (512-sizex)*45/256+(sizex-256)*10/256;
    par.lamada    =   0.18;
    par.siglamda  =   0.67;
    par.Maxgroupsize  =  sigma*1000;
    par.nSig          =  sigma;
    
    if sigma <= 20
        par.patsize      = 7;
        par.tao      = 4.7;
        par.tot_iter = 4;
        load PCLR_para_gmm/PCLR_model7.mat;
    elseif sigma <= 40
        par.patsize      = 8;
        par.tao      = 4.8;
        par.tot_iter = 5;
        load PCLR_para_gmm/PCLR_model8.mat;
    elseif sigma <= 60
        par.patsize       = 9;
        par.tao       = 5.0;
        par.tot_iter  = 5;
        load PCLR_para_gmm/PCLR_model9.mat;
    else
        par.patsize       = 10;
        par.tao       = 5.2;
        par.tot_iter  = 6; 
        load PCLR_para_gmm/PCLR_model10.mat;
    end
    
    par.N   =   h - par.patsize + 1;
    par.M   =   w - par.patsize + 1;
    par.r   =   1:par.N;
    par.c   =   1:par.M; 

end

