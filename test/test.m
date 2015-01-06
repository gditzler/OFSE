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
opts.truncate = 10;
opts.partial_test = 1;

load german

[labels,data] = standardize_data(data);
[mistakes, timerz] = ofs_bagging(data, labels, opts);
opts.partial_test = 0;
[mistakes2, timerz] = ofs_bagging(data, labels, opts);


figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(mean(cumsum(mistakes2(:, 1:end-1)), 2),'b-.')
plot(cumsum(mistakes(:, end)),'r')
plot(cumsum(mistakes2(:, end)),'k')
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
opts.truncate = 10;
opts.partial_test = 1;

load german

[labels,data] = standardize_data(data);
[mistakes, timerz] = ofs_boosting(data, labels, opts);
opts.partial_test = 0;
[mistakes2, timerz] = ofs_boosting(data, labels, opts);


figure;
hold on;
plot(mean(cumsum(mistakes(:, 1:end-1)), 2),'b')
plot(mean(cumsum(mistakes2(:, 1:end-1)), 2),'b-.')
plot(cumsum(mistakes(:, end)),'r')
plot(cumsum(mistakes2(:, end)),'k')
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
