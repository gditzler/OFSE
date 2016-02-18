
clc; 
clear; 
close all; 

addpath('ofse/');
addpath('../thesis-code/feat_sel/FEAST/FEAST/FSToolbox/');
addpath('../thesis-code/feat_sel/FEAST/FEAST/MIToolbox/');
addpath('../thesis-code/utils/');

datasets = {'a8a'
            'german'
            'magic04'
            'spambase'
            'splice'
            'svmguide3'
            'ionosphere'
            'ovariancancer'
            'sido0'
            'miniboone.csv'
            'breast-cancer-wisc-diag.csv'
            'breast-cancer-wisc-prog.csv'
            'chess-krvkp.csv'
            'conn-bench-sonar-mines-rocks.csv'
            'connect-4.csv'
            'molec-biol-promoter.csv'
            'parkinsons.csv'
            'spect_train.csv'};

timers = {};

opts.lambda = 1;          % bagging parameter
opts.ensemble_size = 25;  % number of ofs classifiers
opts.epsilon = 0.2;       % exploration parameter
opts.eta = 0.2;           % learning rate
opts.R = 10;              % regularization
opts.avg = 15;            % # of permuation averages
opts.frac = 0.25;         % fraction of features to select
opts.annel = 1;
fracs = .1:0.02:.3;



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
  
  if strcmp(datasets{nd}, 'ionosphere')
    data(:, 3) = [];
  end
  
  [labels,data] = standardize_data(data);
  data2 = data_binner(data(:, 2:end));
  labels2 = labels;
  labels2(labels2==-1) = 2;
  data2(isnan(data2)) = 1;
  
  
  for k = 1:length(fracs)
    opts.frac = fracs(k);
    opts.truncate = floor(size(data,2)*opts.frac);
    
    tic;
    ofs_boosting(data, labels, opts);
    ts.ofse(k) = toc;
    
    tic;
    feast('mrmr',floor(size(data,2)*opts.frac),data2,labels2);
    ts.mrmr(k) = toc;
    
    tic;
    feast('jmi',floor(size(data,2)*opts.frac),data2,labels2);
    ts.jmi(k) = toc;
    
  end
  
  
  timers{nd} = 1;
  

end
save('mat/time_experiments.mat');

