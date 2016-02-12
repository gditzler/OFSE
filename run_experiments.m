%% run_experiments.m
%  Experiments Implemented
%   1) cumulative mistakes & timing on real-world datasets without
%   annealing
%   2) cumulative mistakes & timing on real-world datasets with annealing
%   3) cumulative mistakes & timing on real-world datasets without
%   annealing. partial information is used for testing as well 
%   4) cumulative mistakes & timing on real-world datasets with annealing.
%   partial information is used for testing as well. 
%% No Annealing
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_no_anneal.mat');

delete(gcp('nocreate'));
%% No Annealing
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.annel = .9999;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_anneal.mat');

delete(gcp('nocreate'));
%% No Annealing (partial information testing)
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.partial = 1;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_no_anneal_partialtest.mat');

delete(gcp('nocreate'));
%% No Annealing (partial information testing)
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.annel = .9999;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_anneal_partialtest.mat');

delete(gcp('nocreate'));
%% No Annealing (_fixedPartial)
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.fixed_partial = true;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_no_anneal_fixedPartial.mat');

delete(gcp('nocreate'));
%% No Annealing (_fixedPartial)
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.annel = .9999;
opts.fixed_partial = false;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
    X = X(:, std(X)~=0);
    Y(Y == 0) = -1;
    data = [Y X];
  else
    load(['data/', datasets{nd}, '.mat'])
    X = data(:, 2:end);
    Y = data(:, 1);
    X = X(:, std(X)~=0);
    data = [Y X];
  end
  
  [labels,data] = standardize_data(data);
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_anneal_fixedPartial.mat');

delete(gcp('nocreate'));
%% No Annealing (partial information testing, _fixedPartial)
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.partial = 1;
opts.fixed_partial = false;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_no_anneal_partialtest_fixedPartial.mat');

delete(gcp('nocreate'));
%% No Annealing (partial information testing, _fixedPartial)
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ... 
  'ionosphere', 'ovariancancer', 'arrhythmia', 'sido0',...
  'miniboone.csv', 'breast-cancer-wisc-diag.csv', 'breast-cancer-wisc-prog.csv', 'chess-krvkp.csv','conn-bench-sonar-mines-rocks.csv',...
  'connect-4.csv','molec-biol-promoter.csv'};
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
opts.annel = .9999;
opts.partial_test = 1;
opts.fixed_partial = false;

% initalize up the cell arrays for results 
mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

% meant for gail's cluster
delete(gcp('nocreate'));
parpool(15);

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
  
  opts.truncate = floor(size(data,2)*opts.frac);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment_anneal_partialtest_fixedPartial.mat');

delete(gcp('nocreate'));
