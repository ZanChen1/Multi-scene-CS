function [par,quantize]=quantize_cell(par,quantize,OMEGA,flag)

if isfield(par,'y')
    if iscell(par.y)==0
        y=cell(size(OMEGA));
        step(1,1)=1;
        step(1,2)=step(1,1)+length(OMEGA{1})-1;
        for i=2:length(OMEGA)
            step(i,1)=step(i-1,2)+1;
            step(i,2)=step(i-1,2)+length(OMEGA{i});
        end
        for i=1:length(step)
            y{i}=par.y(step(i,1):step(i,2));
        end
        par.y=y;
    end
end

if flag==1
    par.bin=cell(size(par.y));
    for i=1:length(par.y)
        
        quantize.Mu(i)=mean(par.y{i}(:));
        quantize.Sigm(i)=std(par.y{i}(:));   
        if size(par.y{i},2)~=0
            [par.bin{i},quantize]=Quantize(par.y{i}(:)',quantize, quantize.Mu(i), quantize.Sigm(i), quantize.bit(i),1);
            par.bin{i}=uint8(par.bin{i})-1;
        end
    end
    
    
elseif flag==0
    
    par.dec=cell(size(par.bin));
    bin=cell(size(par.bin));
    for i=1:length(par.bin)
        bin{i}=par.bin{i}+1;
        [par.dec{i},quantize]=Quantize(bin{i}(:)',quantize, quantize.Mu(i), quantize.Sigm(i), quantize.bit(i),0);
    end
    
    
end



end