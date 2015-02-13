%% plot no annealing
clc;
clear;
close all;

% mistakes_experiment_anneal
% mistakes_experiment_anneal_fixedPartial
% mistakes_experiment_anneal_partialtest
% mistakes_experiment_anneal_partialtest_fixedPartial
% mistakes_experiment_no_anneal
% mistakes_experiment_no_anneal_fixedPartial
% mistakes_experiment_no_anneal_partialtest
% mistakes_experiment_no_anneal_partialtest_fixedPartial
fname = 'mistakes_experiment_no_anneal_partialtest_fixedPartial';

load(['mat/', fname, '.mat']);

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
  
  saveas(h, ['eps/', datasets{nd}, '_', fname, '.eps'], 'eps2c')
  close all;
end
%% print times 
clc;
clear;
close all;

% mistakes_experiment_anneal
% mistakes_experiment_anneal_fixedPartial
% mistakes_experiment_anneal_partialtest
% mistakes_experiment_anneal_partialtest_fixedPartial
% mistakes_experiment_no_anneal
% mistakes_experiment_no_anneal_fixedPartial
% mistakes_experiment_no_anneal_partialtest
% mistakes_experiment_no_anneal_partialtest_fixedPartial
load mat/mistakes_experiment_no_anneal.mat

disp('Data Set & OFS-Bag & OFS-Boo & OFS-Bag-R & OFS-Boo-R')
for nd = 1:length(datasets)
  disp([datasets{nd}, ' & ', num2str(round(1000*timerz_oba{nd})/1000), ' & ', num2str(round(1000*timerz_obo{nd})/1000), ...
    ' & ', num2str(round(1000*timerz_rba{nd})/1000), ' & ', num2str(round(1000*timerz_rbo{nd})/1000)])
end
%% print mistakes 
clc;
clear;
close all;

% mistakes_experiment_anneal
% mistakes_experiment_anneal_fixedPartial
% mistakes_experiment_anneal_partialtest
% mistakes_experiment_anneal_partialtest_fixedPartial
% mistakes_experiment_no_anneal
% mistakes_experiment_no_anneal_fixedPartial
% mistakes_experiment_no_anneal_partialtest
% mistakes_experiment_no_anneal_partialtest_fixedPartial
load mat/mistakes_experiment_no_anneal.mat

disp('Data Set & Single & OFS-Bag & OFS-Boo & OFS-Bag-R & OFS-Boo-R')
for nd = 1:length(datasets)
  if strcmp(datasets{nd}, 'ionosphere')
    load ionosphere
    [~,~,y] = unique(Y);
    y(y==2) = -1;
    X(:, 2) = [];
    data = [y X];
  elseif strcmp(datasets{nd}, 'ovariancancer')
    load ovariancancer
    [~,~,y] = unique(grp);
    y(y==2) = -1;
    data = [y  obs];
  else
    load(['data/', datasets{nd}, '.mat'])
    X = data(:, 2:end);
    Y = data(:, 1);
    X = X(:, std(X)~=0);
    data = [Y X];
  end
  
  disp([datasets{nd}, ' & ', num2str(sum(mean(mistakes_oba{nd}(:,1:end-1),2))/(size(data,1)-1)), ...
    ' & ', num2str(sum(mistakes_oba{nd}(:,end))/(size(data,1)-1)), ' & ', num2str(sum(mistakes_obo{nd}(:,end))/(size(data,1)-1)), ...
    ' & ', num2str(sum(mistakes_rba{nd}(:,end))/(size(data,1)-1)), ' & ', num2str(sum(mistakes_rbo{nd}(:,end))/(size(data,1)-1))])
end
