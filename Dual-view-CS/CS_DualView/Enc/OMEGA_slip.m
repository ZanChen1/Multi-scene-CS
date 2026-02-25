function [q]=OMEGA_slip(K)

	q(1,1)=1;
	q(1,2)=q(1,1)+K(1)-1;
	for i=2:length(K)
		q(i,1)=q(i-1,2)+1;
		q(i,2)=q(i-1,2)+K(i);
	end

end