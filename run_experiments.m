clc; 
clear; 
close all; 

addpath('ofse/');

datasets = {'a8a', 'german', 'magic04', 'spambase', 'splice', 'svmguide3', 'sido0'};

opts.lambda = 1;          % bagging parameter
opts.ensemble_size = 25;  % number of ofs classifiers
opts.epsilon = 0.2;       % exploration parameter
opts.eta = 0.2;           % learning rate
opts.R = 10;              % regularization
opts.truncate = 25;       %
opts.avg = 25;
opts.frac = 0.25;


mistakes_oba = cell(1, length(datasets));
mistakes_obo = cell(1, length(datasets));
mistakes_rba = cell(1, length(datasets));
mistakes_rbo = cell(1, length(datasets));
timerz_oba = cell(1, length(datasets));
timerz_obo = cell(1, length(datasets));
timerz_rba = cell(1, length(datasets));
timerz_rbo = cell(1, length(datasets));

delete(gcp('nocreate'));
parpool(25);

for nd = 1:length(datasets) 
  disp(['Running: ', datasets{nd}]);
  load(['data/',datasets{nd},'.mat'])
  [labels,data] = standardize_data(data);
  
  opts.truncate = floor(size(data,2)/4);
  
  disp('  > OFSE-Bag');
  [mistakes_oba{nd}, timerz_oba{nd}] = ofs_bagging_avg(data, labels, opts);
  
  disp('  > OFSE-Boo');
  [mistakes_obo{nd}, timerz_obo{nd}] = ofs_boosting_avg(data, labels, opts);
  
  disp('  > OFSE-Bag-R');
  [mistakes_rba{nd}, timerz_rba{nd}] = ofs_bagging_randtrunc_avg(data, labels, opts);
  
  disp('  > OFSE-Boo-R');
  [mistakes_rbo{nd}, timerz_rbo{nd}] = ofs_boosting_randtrunc_avg(data, labels, opts);
end
save('mat/mistakes_experiment.mat');

delete(gcp('nocreate'));
