function [gamma_out] = Onsager_2(x_t, m, sigma_hat, denoi)


n = numel(x_t{1});
epsilon = 1;
eta=randn(n,1);
gamma = n/m*eta.*(denoi(x_t{1}+epsilon*eta,sigma_hat)-x_t{2});
gamma_out = repmat(sum(gamma),size(gamma));
gamma_out = gamma_out./n;

end