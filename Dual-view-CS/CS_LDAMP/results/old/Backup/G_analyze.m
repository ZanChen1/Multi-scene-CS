histogram(x_re,x_range,'Normalization','pdf', 'FaceColor', [0.6,0.6,0.6], 'EdgeColor', [1,1,1]);
hold on, 
pdft = ksdensity(x_re, x_range);
plot(x_range, pdft, 'LineWidth', 2, 'Color', [0,0,0])
hold on
pdf_sigm = normpdf(x_range, 0, sigma_x_hat);
plot(x_range, pdf_sigm, 'LineWidth', 2, 'Color', [1,1,0])
hold off
