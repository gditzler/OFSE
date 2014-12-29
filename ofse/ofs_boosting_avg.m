function [mistakes, timerz] = ofs_boosting_avg(data, labels, opts)


mistakes_c= cell(1, opts.avg);
timerz_c = cell(1, opts.avg);
parfor k = 1:opts.avg 
  ks = randperm(numel(labels));
  data_k = data(ks, :);
  labels_k = labels(ks);
  [mistakes_c{k}, timerz_c{k}] = ofs_boosting(data_k, labels_k, opts);
end


mistakes = zeros(size(mistakes_c{1}));
timerz = zeros(size(timerz_c{1}));

for k = 1:opts.avg 
  mistakes = mistakes + mistakes_c{k};
  timerz = timerz + timerz_c{k};
end
mistakes = mistakes/opts.avg;
timerz = timerz/opts.avg;