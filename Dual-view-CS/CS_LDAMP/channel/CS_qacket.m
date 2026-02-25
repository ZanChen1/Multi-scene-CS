function []=packet_lost(PLR)

	load('..\channel\transmit_data.mat');
	N_dc=quantize.N_dc;
	for i=2:length(Trans.data)
		v=Trans.data(i);
		[v_short,N_short,v_lossmark]=setpackloss(v,length(v),quantize.packsize,PLR); %N_short表示除去丢的采样值剩下的采样个数,%v_lossmark为接收到的采样值的标志，1为丢包，0为接收到了		
		[n,m]=find(v_lossmark>0);
		Trans.data{i}(n)=[];
		quantize.OMEGA{i}(n)=[];

	end

end