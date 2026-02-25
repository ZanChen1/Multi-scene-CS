function b = A_bp(x,OMEGA,P_image,P_block,image_height,image_width,block_height,block_width,Phi)

x=x(P_image);
x=reshape(x,[image_height,image_width]);
B=im2col(x,[block_height,block_width],'distinct');
B=B(P_block,:);
% for i=1:size(B,2)
%     B_temp=reshape(B(:,i),[block_width block_height]);
%     B_temp=Phi*B_temp*Phi';
%     fx(:,i)=B_temp(:);
% end
fx=Phi*B;
fx=fx';
if iscell(OMEGA)
    for i=1:length(OMEGA)
        b{i}=fx(OMEGA{i});
    end
else
    b=fx(OMEGA);    %等价于对置乱的结果随机取一部分得到y
end

end