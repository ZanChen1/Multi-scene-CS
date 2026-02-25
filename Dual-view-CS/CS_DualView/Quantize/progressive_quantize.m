function [par]=progressive_quantize(par,quantize)

	l=quantize.refine_layer;
  	OMEGA=quantize.OMEGA;
	A=@(z)A_bp2(z,OMEGA,quantize.P_image,quantize.P_block,quantize.Phi);
    %At=@(z)At_bp2(z,OMEGA,quantize.P_image,quantize.P_block,quantize.Phi);
    par.y=A(par.rim);

    [par,~]=quantize_cell(par,quantize,OMEGA,1);
    [par,~]=quantize_cell(par,quantize,OMEGA,0);

    par.bin{1}=par.or_bin{1};
    par.bin{2}=par.or_bin{2};
    for i=3:length(OMEGA)
     	
    	for j=1:length(par.bin{i})

            t=par.bin{i}(j);
    		temp_y=mod(t,2^l);
    		temp_layer=par.or_bin{i}(j);
    		Dis=abs(temp_layer-temp_y+[-2^l,0,2^l]);
    		S=sign(temp_layer-temp_y+[-2^l,0,2^l]);
            [~,IX]=sort(Dis,'ascend');
            
            if IX(1)<2^(l-1)
                par.bin{i}(j)=t+Dis(IX(1))*S(IX(1));

            elseif IX(1)==2^(l-1)
                if par.y{i}{j}<par.dec{i}{j}
                    par.bin{i}(j)=t-IX(1);
                elseif par.y{i}{j}>par.dec{i}{j}
                    par.bin{i}(j)=t+IX(1);
                end

            end          
                  
            if par.bin{i}(j)>=2^quantize.bit(3)
                par.bin{i}(j)=t-IX(1);
            end
            if par.bin{i}(j)<0
                par.bin{i}(j)=t+IX(1);
            end

        end
		
    end
     %par.bin{3}=par.or_bin{3};
    [par,~]=quantize_cell(par,quantize,OMEGA,0);




end