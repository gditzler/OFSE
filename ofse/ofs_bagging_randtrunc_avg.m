function [mistakes, timerz] = ofs_bagging_randtrunc_avg(data, labels, opts)


mistakes_c= cell(1, opts.avg);
timerz_c = cell(1, opts.avg);
parfor k = 1:opts.avg 
  ks = randperm(numel(labels));
  data_k = data(ks, :);
  labels_k = labels(ks);
  [mistakes_c{k}, timerz_c{k}] = ofs_bagging(data_k, labels_k, ...
    set_opts(opts, size(data,2)));
end


mistakes = zeros(size(mistakes_c{1}));
timerz = zeros(size(timerz_c{1}));

for k = 1:opts.avg 
  mistakes = mistakes + mistakes_c{k};
  timerz = timerz + timerz_c{k};
end
mistakes = mistakes/opts.avg;
timerz = timerz/opts.avg;

function set_opts(opts, K)
opts.truncate = poissrnd(floor(opts.frac*K), 1, opts.ensemble_size+1);
