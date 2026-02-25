


function [v_short,N_short,v_lossmark]=setpackloss(v_bin,N,packsize,PLR)


n=ceil(N/packsize);                       %包的大小和数量
num=randperm(n);
xx=ceil(n*PLR);
lossmark=zeros(n,1);
v_lossmark=zeros(N,1); 
 for i=1:xx
    lossmark(num(i))=1;            %丢包的位置用1表示，未丢包用0表示
 end

  for i=1:N
     if(lossmark(ceil(i/packsize))==1)
         v_lossmark(i)=1;
     end 
 end
 v_short=v_bin;
 N_short=N-xx*packsize;
 if(lossmark(n)==1)                    %因为最后一个包中只有4个采样值，如果最后一个包的丢包标志为真，只能减去4个，而不是减去20个
     N_short=N_short+(n*packsize-N);
 end

 