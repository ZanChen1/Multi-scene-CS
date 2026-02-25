function  par  =  NLR_set_parameters(rate)
par.win        =    6;    % Patch size
par.nblk       =    45;   
par.step       =    min(6, par.win-1);
par.beta       =    0.01;
%-------------------------------------------------------
% The random sampling pattern used in L1-magic software
%-------------------------------------------------------

    par.K0     =   3;
    par.K      =   20;
    par.c0     =   0.49; 
    
    c0=[0.49,1.5];
    c1=[1.8,1.55,1.35,1.32,1.15,0.9,0.75];
    nSig=[4.66,3.25,2.65,2.35,2.1,1.8,1.4];
    
    [~,I]=min(abs(rate-[0.05,0.1,0.15,0.2,0.25,0.3,0.35]));
    
    switch I
        case 1
            par.nSig=4.66;            
            par.c1=1.8;
            par.c0=1.5;
            
        case 2            
            par.nSig=3.25;         
            par.c1=1.55;     
            
        case 3
            par.nSig=2.65;
            par.c1=1.35;
            
        case 4
            par.nSig=2.35;
            par.c1=1.32;
            
        case 5
            par.nSig=2.1;
            par.c1=1.15;
            
        case 6
            par.nSig=1.8;
            par.c1=0.9;
            
        case 7
            par.nSig=1.8;
            par.c1=0.9;
            
    end        
     

return;
