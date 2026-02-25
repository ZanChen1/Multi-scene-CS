function [opts] = opts_set(x,ratio,theta,beta)
opts.Org = x;
[x_row,x_col] = size(x);
opts.mu = 2^8;
opts.beta = 2^beta;
opts.tol = 1E-3;
opts.maxit = 300;
opts.theta = theta;

opts.block_size = 32;
opts.ratio = ratio;
opts.row = x_row;
opts.col = x_col;





end