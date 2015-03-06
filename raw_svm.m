% LIBSVM test

load('C:\data\ucddb\drowsy_awake_7feat.mat')
data = FEAT;
label = CLASS;

% Split Data
train_data = data(1:25000,:);
train_label = label(1:25000,:);
test_data = data(25001:end,:);
test_label = label(25001:end,:);

% Linear Kernel
model_linear = svmtrain(train_data, train_label);
% c -> Parameter C of C-SVC (default 0.5)
% g -> Parameter gamma in kernel function (default 1/num_features)
% t -> Kernel type (default 2, rbf)
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);

% Precomputed Kernel
model_precomputed = svmtrain(train_label, [(1:150)', train_data*train_data'], '-t 4');
[predict_label_P, accuracy_P, dec_values_P] = svmpredict(test_label, [(1:120)', test_data*train_data'], model_precomputed);

accuracy_L % Display the accuracy using linear kernel
accuracy_P % Display the accuracy using precomputed kernel