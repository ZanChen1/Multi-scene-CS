function [gamma_out] = Onsager_patch(x_t, m, sigma_hat, denoi)


n = numel(x_t{1});

size_x = sqrt(n);
iter = 1;
epsilon = 1;
gamma_out = 0;
for i=1:iter
    randn('state',i);
    rand('state',i);
    eta=randn(n,1);
    gamma = n/m*eta.*(denoi(x_t{1}+epsilon*eta,sigma_hat)-x_t{2});
    gamma = reshape(gamma,[size_x size_x]);
    func=@(z) mean_replace(z);
    gamma_out = gamma_out + qt_function(gamma, 1, func);
    gamma_inter1 = gamma_out(128:129,128:129)./i;
    gamma_inter2 = mean(gamma_inter1(:));
end
gamma_out = gamma_inter2;
end

function x_out = qt_split(x)

x_out ={};
[n,m]  = size(x);
n = round(n/2);
m = round(m/2);
x_out{1,1} = x(1:n,1:m);
x_out{2,1} = x(n+1:end,1:m);
x_out{1,2} = x(1:n,m+1:end);
x_out{2,2} = x(n+1:end,m+1:end);

end

function x = qt_comb(x)

x = [x{1,1}, x{1,2}; x{2,1}, x{2,2}];

end

function x_out = qt_function(x, level, func)

x_temp = qt_split(x);
for i=1:2
    for j=1:2
        x_temp{i,j} = func(x_temp{i,j});
    end
end

x_out = qt_comb(x_temp);

%x_out = func(x);

end


function [x] = mean_replace(x)

x_temp = ones(size(x));
x= x_temp.*mean(x(:));

end

