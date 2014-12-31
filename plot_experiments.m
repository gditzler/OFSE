%% plot no annealing
clc;
clear;
close all;

load mat/mistakes_experiment_no_anneal.mat

fs = 20;
lw = 2;

for nd = 1:length(datasets)
  h = figure;
  hold on;
  box on;
  
  mistakes = mistakes_oba{nd};
  plot(mean(cumsum(mistakes(:, 1:end-1)), 2), 'c', 'LineWidth', lw)
  plot(cumsum(mistakes(:, end)), 'r', 'LineWidth', lw)

  mistakes = mistakes_obo{nd};
  plot(cumsum(mistakes(:, end)), 'k', 'LineWidth', lw)
  
  mistakes = mistakes_rba{nd};
  plot(cumsum(mistakes(:, end)), 'b', 'LineWidth', lw)
  
  mistakes = mistakes_rbo{nd};
  plot(cumsum(mistakes(:, end)), 'm', 'LineWidth', lw)
  axis tight;
  legend('Single', 'OFS-Bag', 'OFS-Boo', 'OFS-Bag-R', 'OFS-Boo-R', 'Location', 'Best')
  set(gca, 'fontsize', fs)
  xlabel('time', 'FontSize', fs)
  ylabel('mistakes', 'FontSize', fs)
  
  saveas(h, ['eps/', datasets{nd},'_mistakes_no_annealing.eps'], 'eps2c')
  close all;
end
%% print times 
clc;
clear;
close all;

load mat/mistakes_experiment_no_anneal.mat

disp('Data Set & OFS-Bag & OFS-Boo & OFS-Bag-R & OFS-Boo-R')
for nd = 1:length(datasets)
  disp([datasets{nd}, ' & ', num2str(timerz_oba{nd}), ' & ', num2str(timerz_obo{nd}), ...
    ' & ', num2str(timerz_rba{nd}), ' & ', num2str(timerz_rbo{nd})])
end
%% plot mistake RATE no annealing
clc;
clear;
close all;

load mat/mistakes_experiment_no_anneal.mat

fs = 20;
lw = 2;

for nd = 1:length(datasets)
  h = figure;
  hold on;
  box on;
  
  mistakes = mistakes_oba{nd};
  plot(mean(cumsum(mistakes(:, 1:end-1)), 2)'./(1:length(mistakes(:, end))), 'c', 'LineWidth', lw)
  plot(cumsum(mistakes(:, end))'./(1:length(mistakes(:, end))), 'r', 'LineWidth', lw)

  mistakes = mistakes_obo{nd};
  plot(cumsum(mistakes(:, end))'./(1:length(mistakes(:, end))), 'k', 'LineWidth', lw)
  
  mistakes = mistakes_rba{nd};
  plot(cumsum(mistakes(:, end))'./(1:length(mistakes(:, end))), 'b', 'LineWidth', lw)
  
  mistakes = mistakes_rbo{nd};
  plot(cumsum(mistakes(:, end))'./(1:length(mistakes(:, end))), 'm', 'LineWidth', lw)
  axis tight;
  legend('Single', 'OFS-Bag', 'OFS-Boo', 'OFS-Bag-R', 'OFS-Boo-R', 'Location', 'Best')
  set(gca, 'fontsize', fs)
  xlabel('time', 'FontSize', fs)
  ylabel('mistakes', 'FontSize', fs)
  
  saveas(h, ['eps/', datasets{nd},'_mistakerate_no_annealing.eps'], 'eps2c')
  close all;
end
