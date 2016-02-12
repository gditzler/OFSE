clc;
clear;
close all;


datasets = {'acute-inflammation.csv'
            'acute-nephritis.csv'
            'adult_train.csv'
            'balloons.csv'
            'bank.csv'
            'blood.csv'
            'breast-cancer-wisc-diag.csv'
            'breast-cancer-wisc-prog.csv'
            'breast-cancer-wisc.csv'
            'chess-krvkp.csv'
            'congressional-voting.csv'
            'conn-bench-sonar-mines-rocks.csv'
            'connect-4.csv'
            'fertility.csv'
            'heart-hungarian.csv'
            'hepatitis.csv'
            'ionosphere.csv'
            'magic.csv'
            'mammographic.csv'
            'miniboone.csv'
            'molec-biol-promoter.csv'
            'parkinsons.csv'
            'pima.csv'
            'spect_train.csv'
            'statlog-australian-credit.csv'
            'statlog-german-credit.csv'
            'statlog-heart.csv'
            'tic-tac-toe.csv'
            'titanic.csv'};


n_data = length(datasets);


for n = 2:n_data
  data = load(['~/Git/ClassificationDatasets/csv/', datasets{n}]);
  labels = data(:, end);
  data = data(:, 1:end);
  disp(['  ', datasets{n}, ':  N', num2str(size(data,1)),  ':  F', num2str(size(data,2))]);
end





