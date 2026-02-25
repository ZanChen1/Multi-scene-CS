function x = gaussian_noise(x, mean, var)

x = double(x);
x = x+normrnd(mean, var, size(x));

end