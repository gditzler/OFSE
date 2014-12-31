clc
clear
close all


K = 100;
X = 0:100; 
lw = 2;
fs = 20;

h = figure; 
hold on;
box on;
plot(X, poisspdf(X, 1), 'r', 'LineWidth', lw);
plot(X, poisspdf(X, 10), 'c', 'LineWidth', lw);
plot(X, poisspdf(X, 25), 'b', 'LineWidth', lw);
plot(X, poisspdf(X, 50), 'k', 'LineWidth', lw);
legend('\lambda=1', '\lambda=10', '\lambda=25', '\lambda=50', 'Location', 'best');
xlabel('B', 'FontSize', fs);
ylabel('pdf', 'FontSize', fs);
set(gca, 'fontsize', fs);
saveas(h, '../eps/poisson_distributions.eps', 'eps2c')