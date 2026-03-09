function b = A_bp2(x,OMEGA,P_image,P_block,Phi)


N=length(P_block);
block_size=sqrt(N);
n=length(P_image)/(block_size^2);
fx=zeros(N,n);
x=x(P_image);
%x=reshape(x,[256 256]);
x=reshape(x,[128 128]);
%x=reshape(x,[512 512]);
B=im2col(x,[block_size block_size],'distinct');
B=B(P_block,:);
for i=1:size(B,2)
    B_temp=reshape(B(:,i),[block_size,block_size]);
    B_temp=Phi*B_temp*Phi';
    fx(:,i)=B_temp(:);
end
%fx=Phi*B;
fx=fx';
if iscell(OMEGA)
    for i=1:length(OMEGA)
        b{i}=fx(OMEGA{i});
    end
else
    b=fx(OMEGA);
end

end