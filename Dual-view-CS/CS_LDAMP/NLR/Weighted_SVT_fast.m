function  [X W]   =   Weighted_SVT_fast( Y, c1, nsig2, U0, flag, c0 )
c1                =   c1*sqrt(2);
n                 =   sqrt(length(U0));
U0                =   reshape(U0, n, n);

A                 =   U0'*Y;
Sigma0            =   sqrt( sum(A.^2, 2) );
V0                =   (diag(1./Sigma0)*A)';

if flag==1
    S                 =   max( Sigma0.^2/size(Y, 2) - 0*nsig2, 0 );
    thr               =   c1*nsig2./ ( sqrt(S) + eps );
    S                 =   soft(Sigma0, thr);
else
    S                 =   soft(Sigma0, c0*nsig2);
end
r                 =     sum( S>0 );
P                 =     find(S);
X                 =     U0(:,P)*diag(S(P))*V0(:,P)';

if r==size(Y,1)
    wei           =     1/size(Y,1);  % 1;
else
    wei           =     (size(Y,1)-r)/size(Y,1);  %  1;
end
W                 =     1*ones( size(X) );
X                 =     X*1;
end
