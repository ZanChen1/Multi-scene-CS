function   [dim, wei, U_arr]  =  Low_rank_appro(nim, par, blk_arr, U_arr, it, flag)
b            =   par.win;
[h  w ch]    =   size(nim);
N            =   h-b+1;
M            =   w-b+1;
r            =   [1:N];
c            =   [1:M];

[X]           =   Im2Patch( nim, par );
Ys           =   zeros( size(X) );
W            =   zeros( size(X) );
L            =   size(blk_arr,2);
T            =   4;
for  i  =  1 : L
    B          =   X(:, blk_arr(:, i));

    if it==1 || mod(it, T)==0
        [tmp_y, tmp_w, U_arr(:,i)]   =   Weighted_SVT( double(B), par.c1, par.nSig^2, flag, par.c0 );
    else
        [tmp_y, tmp_w]   =   Weighted_SVT_fast( double(B), par.c1, par.nSig^2, U_arr(:,i), flag, par.c0 );
    end

    Ys(:, blk_arr(:,i))   =   Ys(:, blk_arr(:,i)) + tmp_y;
    W(:, blk_arr(:,i))    =   W(:, blk_arr(:,i)) + tmp_w;
end

dim     =  zeros(h,w);
wei     =  zeros(h,w);
k       =   0;
for i  = 1:b
    for j  = 1:b
        k    =  k+1;
        dim(r-1+i,c-1+j)  =  dim(r-1+i,c-1+j) + reshape( Ys(k,:)', [N M]);
        wei(r-1+i,c-1+j)  =  wei(r-1+i,c-1+j) + reshape( W(k,:)', [N M]);
    end
end



end

