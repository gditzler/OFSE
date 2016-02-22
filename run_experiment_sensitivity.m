clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv', 'parkinsons.csv', 'spect_train.csv'};
% datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ...
%   'sido0', 'uni_S10000F500R25', 'uni_S10000F500R50', 'uni_S10000F500R100', ...
%   'uni_S10000F500R150', 'uni_S10000F500R200'};


opts.lambda = 1;          % bagging parameter
opts.ensemble_size = 25;  % number of ofs classifiers
opts.epsilon = 0.2;       % exploration parameter
opts.eta = 0.2;           % learning rate
opts.R = 10;              % regularization
opts.avg = 15;            % # of permuation averages
opts.frac = 0.25;         % fraction of features to select
opts.annel = 1;
fracs = .1:0.02:.3;

% initalize up the cell arrays for results 
mistakes_oba = zeros(length(fracs), length(datasets));
mistakes_obo = zeros(length(fracs), length(datasets));
mistakes_rba = zeros(length(fracs), length(datasets));
mistakes_rbo = zeros(length(fracs), length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(opts.avg);

for nd = 1:length(datasets) 
  disp(['Running: ', datasets{nd}]);

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
  [labels,data] = standardize_data(data);
  
  
  for n = 1:length(fracs)
    opts.frac = fracs(n);
    opts.truncate = floor(size(data,2)*opts.frac);
  
    disp('  > OFSE-Bag');
    mistakes = cumsum(ofs_bagging_avg(data, labels, opts));
    mistakes_oba(n, nd) =  mistakes(end, end);
  
    disp('  > OFSE-Boo');
    mistakes = ofs_boosting_avg(data, labels, opts);
    mistakes_obo(n,nd) = cumsum(mistakes);
    
    disp('  > OFSE-Bag-R');
    mistakes = ofs_bagging_randtrunc_avg(data, labels, opts);
    mistakes_rba(n,nd) = cumsum(mistakes);
    
    disp('  > OFSE-Boo-R');
    mistakes = ofs_boosting_randtrunc_avg(data, labels, opts);
    mistakes_rbo(n,nd) = cumsum(mistakes);
  end
end
save('mat/B_sensitivity.mat');

delete(gcp('nocreate'));
