sigma_summary = dlmread(['sigma_summary.csv']); 

sigma_mean = mean(sigma_summary,1);
sigma_sig = std(sigma_summary, 1, 1);

itr1 = 1:2:length(sigma_mean);
itr2 = 2:2:length(sigma_mean);

figure(1),plot(sigma_mean, '-')
hold on
plot(itr1, sigma_mean(itr1), '.')
hold on
plot(itr2, sigma_mean(itr2), 'o')
ca = gca;
ca.YLim = [0, 500];
ca.XLim = [0, 23];
hold on
errorbar(sigma_mean, sigma_sig)

hold off

% figure(2),plot(error_summary, '-o')
% ca = gca;
% ca.YLim = [0, 20];
% ca.XLim = [0, 23];

