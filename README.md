# code_matlab
Matlab codes for pnne.
Written with Matlab 2024b.

Please see https://pnnehome.github.io/ for documenation.

The following is an example of using the pre-trained model on an example dataset.

clear

% load pre-trained model<br>
load('trained_nne.mat')

% load data<br>
load('./sample_data/data_Expedia_Kaggle1.mat')

% obtain estimate<br>
result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx);

disp(result)

% obtain estimate and standard error<br>
result = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, se = true);

disp(result)

<br><br>

w/o readme: f763; w/o readme and sample_data: fe81
