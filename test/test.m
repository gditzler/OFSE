%% online bagging + ofs
clc;
clear;
close all;

addpath('../ofse/');
addpath('../data/');

opts.lambda = 1;
opts.ensemble_size = 10;
opts.epsilon = 0.2;
opts.eta = 0.2;
opts.R = 10;
opts.truncate = 25;

load a8a

[labels,data] = standardize_data(data);
[mistakes, error_counts, error_counts_idx, timerz] = ofs_bagging(data, labels, opts);


figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(cumsum(mistakes(:, end)),'r')

figure;
hold on;
semilogx(error_counts_idx, mean(error_counts(:, 1:end-1), 2),'b')
semilogx(error_counts_idx, error_counts(:, end), 'r')
%% online boosting + ofs
clc;
clear;
close all;

addpath('../ofse/');
addpath('../data/');

opts.lambda = 1;
opts.ensemble_size = 10;
opts.epsilon = 0.2;
opts.eta = 0.2;
opts.R = 10;
opts.truncate = 25;

load german

[labels,data] = standardize_data(data);
[mistakes, error_counts, error_counts_idx, timerz] = ofs_boosting(data, labels, opts);


figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(cumsum(mistakes(:, end)),'r')

figure;
hold on;
semilogx(error_counts_idx, mean(error_counts(:, 1:end-1), 2),'b')
semilogx(error_counts_idx, error_counts(:, end), 'r')

%% parallel 
clc
clear
close all

addpath('../ofse/');
addpath('../data/');

opts.lambda = 1;
opts.ensemble_size = 10;
opts.epsilon = 0.2;
opts.eta = 0.2;
opts.R = 10;
opts.truncate = 12;
opts.avg = 5;

load german

delete(gcp('nocreate'));
parpool(4);

[labels,data] = standardize_data(data);
[mistakes, error_counts, error_counts_idx, timerz] = ofs_bagging_avg(data, labels, opts);

figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(cumsum(mistakes(:, end)),'r')

figure;
hold on;
semilogx(2.^error_counts_idx(1:end-1), mean(error_counts(:, 1:end-1), 2),'b')
semilogx(2.^error_counts_idx(1:end-1), error_counts(:, end), 'r')
delete(gcp('nocreate'));
