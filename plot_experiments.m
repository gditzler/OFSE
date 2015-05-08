%% plot no annealing
clc;
clear;
close all;

% fname = 'mistakes_experiment_anneal';
% fname = 'mistakes_experiment_anneal_fixedPartial';
% fname = 'mistakes_experiment_anneal_partialtest';
% fname = 'mistakes_experiment_anneal_partialtest_fixedPartial';
fname = 'mistakes_experiment_no_anneal';
% fname = 'mistakes_experiment_no_anneal_fixedPartial';
% fname = 'mistakes_experiment_no_anneal_partialtest';
% fname = 'mistakes_experiment_no_anneal_partialtest_fixedPartial';

addpath('ofse/');

load(['mat/', fname, '.mat']);

fs = 20;
lw = 2;

for nd = 1:length(datasets)
  h = figure;
  hold on;
  box on;
  
  mistakes = mistakes_oba{nd};
  plot(mean(csum(mistakes(:, 1:end-1)), 2), 'c', 'LineWidth', lw)
  plot(csum(mistakes(:, end)), 'r', 'LineWidth', lw)

  mistakes = mistakes_obo{nd};
  plot(csum(mistakes(:, end)), 'k', 'LineWidth', lw)
  
  mistakes = mistakes_rba{nd};
  plot(csum(mistakes(:, end)), 'b', 'LineWidth', lw)
  
  mistakes = mistakes_rbo{nd};
  plot(csum(mistakes(:, end)), 'm', 'LineWidth', lw)
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
  
  pmat = [sum(mean(mistakes_oba{nd}(:,1:end-1),2))/(size(data,1)-1) ...
    sum(mistakes_oba{nd}(:,end))/(size(data,1)-1) ...
    sum(mistakes_obo{nd}(:,end))/(size(data,1)-1) ...
    sum(mistakes_rba{nd}(:,end))/(size(data,1)-1) ...
    sum(mistakes_rbo{nd}(:,end))/(size(data,1)-1)];
  psort = zeros(1,5);
  [~, ps] = sort(pmat);
  for i = 1:5, psort(i) = find(ps == i); end
  
  disp([datasets{nd}, ' & ', num2str(pmat(1)), ' (', num2str(psort(1)), ')' ...
    ' & ', num2str(pmat(2)), ' (', num2str(psort(2)), ')' ...
    ' & ', num2str(pmat(3)), ' (', num2str(psort(3)), ')' ...
    ' & ', num2str(pmat(4)), ' (', num2str(psort(4)), ')' ...
    ' & ', num2str(pmat(5)), ' (', num2str(psort(5)), ') \\']);
end
