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

load german

[labels,data] = standardize_data(data);
[mistakes, timerz] = ofs_bagging(data, labels, opts);


figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(cumsum(mistakes(:, end)),'r')
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
[mistakes, timerz] = ofs_boosting(data, labels, opts);


figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(cumsum(mistakes(:, end)),'r')
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
[mistakes, timerz] = ofs_bagging_avg(data, labels, opts);

figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(cumsum(mistakes(:, end)),'r')
delete(gcp('nocreate'));
