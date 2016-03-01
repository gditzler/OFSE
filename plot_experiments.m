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
  if isempty(strfind(datasets{nd}, 'csv'))
    name = datasets{nd};
  else
    name = strrep(datasets{nd},'.csv','');
  end
  if strcmp(name, 'arrhythmia')
    continue;
  end
  
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
  
  
  
  saveas(h, ['eps/', name, '_', fname, '.eps'], 'eps2c')
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
  if isempty(strfind(datasets{nd}, 'csv'))
    name = datasets{nd};
  else
    name = strrep(datasets{nd},'.csv','');
  end
  if strcmp(name, 'arrhythmia')
    continue;
  end
  disp([name, ' & ', num2str(round(1000*timerz_oba{nd})/1000), ' & ', num2str(round(1000*timerz_obo{nd})/1000), ...
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
load mat/mistakes_experiment_no_anneal
% mistakes_experiment_no_anneal_fixedPartial
% mistakes_experiment_no_anneal_partialtest
%load mat/mistakes_experiment_no_anneal_partialtest_fixedPartial
%load mat/mistakes_experiment_no_anneal.mat
psorts = [];
disp('Data Set & Single & OFS-Bag & OFS-Boo & OFS-Bag-R & OFS-Boo-R')
for nd = 1:length(datasets)
  if isempty(strfind(datasets{nd}, 'csv'))
    name = datasets{nd};
  else
    name = strrep(datasets{nd},'.csv','');
  end
  if strcmp(name, 'arrhythmia')
    continue;
  end
  
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
  elseif strcmp(datasets{nd}, 'arrhythmia')
    load arrhythmia
    dels = find(Y==16);
    Y(dels) = [];
    X(dels, :) = [];
    X(:, [11,2,14]) = [];
    z = sum(isnan(X),2);
    X(z==1, :) = [];
    Y(z==1) = [];
    Y(Y~=1) = -1;
    data = [Y X];
    clear dels Description VarNames X Y z
  elseif length(findstr('csv', datasets{nd})) > 0
    data = load(['../ClassificationDatasets/csv/', datasets{nd}]);
    X = data(:, 1:end-1);
    Y = data(:, end);
    Y(Y == 0) = -1;
    X = X(:, std(X)~=0);
    data = [Y X];
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
  
  disp([name, ' & ', num2str(pmat(1)), ' (', num2str(psort(1)), ')' ...
    ' & ', num2str(pmat(2)), ' (', num2str(psort(2)), ')' ...
    ' & ', num2str(pmat(3)), ' (', num2str(psort(3)), ')' ...
    ' & ', num2str(pmat(4)), ' (', num2str(psort(4)), ')' ...
    ' & ', num2str(pmat(5)), ' (', num2str(psort(5)), ') \\']);
  psorts = [psorts; psort];
end

[N,k] = size(psorts);
R = mean(psorts);
alpha = .1;

chi2 = (12*N)/(k*(k+1))*(sum(R.^2)-k*(k+1)^2/4);
Ff = (N-1)*chi2/(N*(k-1)-chi2);

z = zeros(k,k);
for j = 1:k
  for i = 1:k
    z(j,i) = (R(j)-R(i))/(sqrt(k*(k+1)/(6*N)));
  end
end
pr = normcdf(-z);
pl = normcdf(z);
p2 = 2*normcdf(-abs(z));
pF = 1 - fcdf(Ff,k-1,(k-1)*(N-1)); % pvalue for the f-test
H = pl < alpha/k;
%% print mistakes 
clc;
clear;
close all;

% mistakes_experiment_anneal
% mistakes_experiment_anneal_fixedPartial
% mistakes_experiment_anneal_partialtest
% mistakes_experiment_anneal_partialtest_fixedPartial
load mat/mistakes_experiment_no_anneal
% mistakes_experiment_no_anneal_fixedPartial
% mistakes_experiment_no_anneal_partialtest
% mistakes_experiment_no_anneal_partialtest_fixedPartial
%load mat/mistakes_experiment_no_anneal.mat
psorts = [];
disp('Data Set & Single & OFS-Bag & OFS-Boo & OFS-Bag-R & OFS-Boo-R')
for nd = 1:length(datasets)
  if isempty(strfind(datasets{nd}, 'csv'))
    name = datasets{nd};
  else
    name = strrep(datasets{nd},'.csv','');
  end
  if strcmp(name, 'arrhythmia')
    continue;
  end
  
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
  elseif strcmp(datasets{nd}, 'arrhythmia')
    load arrhythmia
    dels = find(Y==16);
    Y(dels) = [];
    X(dels, :) = [];
    X(:, [11,2,14]) = [];
    z = sum(isnan(X),2);
    X(z==1, :) = [];
    Y(z==1) = [];
    Y(Y~=1) = -1;
    data = [Y X];
    clear dels Description VarNames X Y z
  elseif length(findstr('csv', datasets{nd})) > 0
    data = load(['../ClassificationDatasets/csv/', datasets{nd}]);
    X = data(:, 1:end-1);
    Y = data(:, end);
    Y(Y == 0) = -1;
    X = X(:, std(X)~=0);
    data = [Y X];
  else
    load(['data/', datasets{nd}, '.mat'])
    X = data(:, 2:end);
    Y = data(:, 1);
    X = X(:, std(X)~=0);
    data = [Y X];
  end
  
  disp([name, ' & ', num2str(size(data,1)), ' & ', num2str(size(data(:,2:end), 2)), ' \\ '])
  %disp([name, ' & ', num2str(size(data,1)), ' & ', num2str(size(data(:,2:end), 2)), ' & ', ...
  %  num2str(100*sum(data(:,1)==1)/numel(data(:,1))), ' & ', num2str(100*sum(data(:,1)==-1)/numel(data(:,1)))])
end

