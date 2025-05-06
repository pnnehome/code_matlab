% this is an example code for using the pre-trained model to estimate
% please check out https://pnnehome.github.io/index.html for detailed
% descriptions


clear

% load pre-trained model
load('trained_nne.mat')

% load data
load('./sample_data/data_Expedia_Kaggle1.mat')

% obtain estimate
result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx);

disp(result)

% obtain estiamte and standard error
result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = true);

disp(result)