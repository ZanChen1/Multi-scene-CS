function  [X W U]   =   Weighted_SVT( Y, c1, nsig2, flag, c0 )
c1                =   c1*sqrt(2);
[U0,Sigma0,V0]    =   svd(full(Y),'econ');
Sigma0            =   diag(Sigma0);
if flag==1
    S                 =   max( Sigma0.^2/size(Y, 2), 0 );
    thr               =   c1*nsig2./ ( sqrt(S) + eps ); %%门限自适应的
    S                 =   soft(Sigma0, thr);
else  % use nuclear norm
     S                 =   soft(Sigma0, c0*nsig2);
end
r                 =   sum( S>0 );
U                 =   U0(:,1:r);
V                 =   V0(:,1:r);
X                 =   U*diag(S(1:r))*V';
% Weighted the reconstructed patches using the weights computed using the
% matrix ranks slightly improve the final results (less than 0.2 dB)
if r==size(Y,1)
    wei           =   1/size(Y,1);   % 1;
else
    wei           =   (size(Y,1)-r)/size(Y,1);  % 1;
end
W                 =   1*ones( size(X) );
X                 =   X*1;
U                 =   U0(:);
end

