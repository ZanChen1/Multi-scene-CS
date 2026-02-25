
A = rand(1000);
B = rand(1000);
profile on;

C = A * B;

profile off;
profreport;

% disp(['Matrix multiplication took ' num2str(flops) ' flops.']);