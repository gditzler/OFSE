%% run_experiments.m
%  Experiments Implemented
%   1) cumulative mistakes & timing on real-world datasets without
%   annealing
%   2) cumulative mistakes & timing on real-world datasets with annealing
%% No Annealing
% Evaluate the OFS ensemble (OFSE) based approaches on several real-world
% data sets. There are four different variations that we are experimenting
% with.
clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', ...
  'sido0', 'uni_S10000F500R25', 'uni_S10000F500R50', 'uni_S10000F500R100', ...
  'uni_S10000F500R150', 'uni_S10000F500R200'};

opts.lambda = 1;          % bagging parameter
opts.ensemble_size = 25;  % number of ofs classifiers
opts.epsilon = 0.2;       % exploration parameter
opts.eta = 0.2;           % learning rate
opts.R = 10;              % regularization
opts.avg = 25;            % # of permuation averages
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
parpool(25);

for nd = 1:length(datasets) 
  disp(['Running: ', datasets{nd}]);
  
  load(['data/',datasets{nd},'.mat'])
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
  'sido0', 'uni_S10000F500R25', 'uni_S10000F500R50', 'uni_S10000F500R100', ...
  'uni_S10000F500R150', 'uni_S10000F500R200'};

opts.lambda = 1;          % bagging parameter
opts.ensemble_size = 25;  % number of ofs classifiers
opts.epsilon = 0.2;       % exploration parameter
opts.eta = 0.2;           % learning rate
opts.R = 10;              % regularization
opts.avg = 25;            % # of permuation averages
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
parpool(25);

for nd = 1:length(datasets) 
  disp(['Running: ', datasets{nd}]);
  
  load(['data/',datasets{nd},'.mat'])
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