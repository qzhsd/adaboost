% % % save the data in .mat
% train_img_file = '../data/train_set/train-images-idx3-ubyte';
% train_label_file = '../data/train_set/train-labels-idx1-ubyte';
% test_img_file = '../data/test_set/t10k-images-idx3-ubyte';
% test_label_file = '../data/test_set/t10k-labels-idx1-ubyte';
% [train_imgs, train_labels] = readMNIST(train_img_file, train_label_file, 20000, 0);
% [test_imgs, test_labels] = readMNIST(test_img_file, test_label_file, 10000, 0);
% save('../data/data.mat', 'train_imgs', 'test_imgs', 'train_labels', 'test_labels');

data_path = './data/data.mat';
load(data_path);

% initialization
% final_scores: N * 10
max_iter = 250;
final_scores = [];
% define weak learners
f_d = size(train_imgs, 2);
D = repmat(1 : f_d, 2*51, 1);
T = repmat((0: 1.0/50 : 1)', 1, f_d * 2);
polarity = repmat([ones(51, 1); (-1)*ones(51, 1)], 1, f_d);
U = [D(:), T(:), polarity(:)];


% adaboost, 10 binary classifiers for each digit
for i = 1 : 10
%     ind_pos = find(train_labels == i-1);
%     ind_nega = find(train_labels ~= i-1);
%     tmp = randperm(length(ind_nega), length(ind_pos));
%     ind_i = [ind_pos; ind_nega(tmp)];
%     t_train_imgs = train_imgs(ind_i, :);
%     t_train_labels = train_labels(ind_i);
%     
%     ind_pos = find(test_labels == i-1);
%     ind_nega = find(test_labels ~= i-1);
%     tmp = randperm(length(ind_nega), length(ind_pos));
%     ind_i = [ind_pos; ind_nega(tmp)];
%     t_test_imgs = test_imgs(ind_i, :);
%     t_test_labels = test_labels(ind_i);
    msg = ['classifier for digit ', num2str(i-1)];
    disp(msg);
    [alphas, ws, train_errors, final_test_score, test_errors, largest_weights, margins] = adaboost(train_imgs, train_labels, test_imgs, test_labels, i-1, U, max_iter);
    result_path = ['./results/', 'digit', num2str(i-1)];
    save(result_path, 'alphas', 'ws', 'train_errors', 'final_test_score', 'test_errors', 'largest_weights', 'margins');
    clear alpha_t w_t train_errors final_test_score test_errors largest_weights margins
end

% final classifier and test errors
% final_scores = [final_scores, final_test_score];
% [~, inds] = max(final_scores, [], 2);
% inds = inds - 1;
% final_error = 1 - sum(inds == test_labels) * 1.0 / length(test_labels)

% result visualization
% plot train/test errors, margins, index of the largest weight, three
% heaviest weights, weak learners




