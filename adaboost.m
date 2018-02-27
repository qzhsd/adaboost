function [alphas, ws, train_errors, final_test_score, test_errors, largest_weights, margins] = adaboost(train_imgs, train_labels, test_imgs, test_labels, digit, U, max_iter)
% INPUTS:
% train_imgs, test_imgs: N * d
% train_labels, test_labels: N*1
% U: (d*t*2) * 3

% OUTPUTS:
% alphas: weak learner, max_iter * 3
% ws: step_size, max_iter * 1
% train_errors: max_iter * 1
% test_errors: max_iter * 1
% final_test_score: N * 1
% largest_weights: max_iter * 1
% margins: N * 5

[N, D] = size(train_imgs);
[N_test, D_test] = size(test_imgs);

tmp = ones(size(train_labels));
tmp(train_labels ~= digit) = -1;
train_labels = tmp;
tmp = ones(size(test_labels));
tmp(test_labels ~= digit) = -1;
test_labels = tmp;

alphas = zeros(max_iter, 3);
ws = zeros(max_iter, 1);
g_t_train = zeros(N, 1);
train_errors = zeros(max_iter, 1);
g_t_test = zeros(N_test, 1);
test_errors = zeros(max_iter, 1);

largest_weights = zeros(max_iter, 1);
margins = [];


for t = 1 : max_iter
    % dataset reweighting
    sample_weights = exp(-train_labels .* g_t_train);
    
    % weak learner selection alpha_t
    % alpha_t: 1*3, y_predict: N*1
    [alpha_t, train_pred] = select_weak_learner(train_imgs, train_labels, U, sample_weights);
    
    % compute the step size w_t
    theta = sum(sample_weights(train_labels ~= train_pred)) / sum(sample_weights);
    w_t = 0.5 * log((1 - theta) / theta);
    
    % update the learned function
    alphas(t, :) = alpha_t;
    ws(t) = w_t;
    
    % compute train_error
    g_t_train = g_t_train + w_t * train_pred;
    train_errors(t) = 1 - sum(sign(g_t_train) == train_labels) * 1.0 / length(train_labels);
    
    % compute test_error
    test_pred = ones(size(test_imgs, 1), 1) * alpha_t(3);
    test_pred(test_imgs(:, alpha_t(1)) < alpha_t(2)) = -alpha_t(3);
    g_t_test = g_t_test + w_t * test_pred;
    test_errors(t) = 1 - sum(sign(g_t_test) == test_labels) * 1.0 / length(test_labels);
    
    % store the largest weights
    [~, largest_weights(t)] = max(sample_weights);
    
    % compute margins if t = 5, 10, 50, 100, 250
    if (t==5) || (t==10) || (t==50) || (t==100) || (t==250)
        tmp = train_labels .* g_t_train;
        margins = [margins, tmp];
    end
    msg = ['iteration: ', num2str(t), ' train_error: ', num2str(train_errors(t)), ' test_error: ', num2str(test_errors(t))];
    disp(msg);
end
final_test_score = g_t_test;
end