% result visualization
% plot train/test errors, margins, index of the largest weight, three
% heaviest weights, weak learners
data_path = './data/data.mat';
load(data_path);

final_scores = [];
for i = 0 : 9
    result_path = ['./results/', 'digit', num2str(i)];
    load(result_path);
    
    % plot train/test errors
    figure(1);
    plot(train_errors, 'LineWidth', 2);
    hold on
    plot(test_errors, 'LineWidth', 2);
    xlim([0 length(train_errors)]);
    ylim([0 0.3]);
    hold off
    legend('train', 'test')
    xlabel('number of iteration');
    ylabel('classification error');
    title(['digit ', num2str(i)]);
    fig_path = ['./results/poe', num2str(i)];
    saveas(gcf, fig_path, 'epsc');
    
    % margins (t=5, 10, 50, 100, 250): N*5
    figure(2);
    for k = 1 : size(margins, 2)
        % [counts, edges] = histcounts(margins(:, k));
        [f,x] = ecdf(margins(:, k));
        plot(x, f, 'LineWidth', 2);
        hold on
    end
    hold off
    legend('t=5', 't=10', 't=50', 't=100', 't=250');
    xlabel('margin');
    ylabel('cumulative distribution function (cdf)');
    title(['digit ', num2str(i)]);
    fig_path = ['./results/margin', num2str(i)];
    saveas(gcf, fig_path, 'epsc');
    
    % index of the largest weight
    figure(3);
    plot(largest_weights, 'LineWidth', 2);
    xlabel('number of iteration');
    ylabel('index of the largest weight');
    title(['digit ', num2str(i)]);
    fig_path = ['./results/largest_weight', num2str(i)];
    saveas(gcf, fig_path, 'epsc');
    
    % three heaviest weights
    edges = unique(largest_weights);
    [n, ~] = histcounts(largest_weights, [edges; edges(end)+1]);
    [~,idx] = sort(n, 'descend');
    
    figure(4);
    subplot(1, 3, 1);
    imshow(reshape(train_imgs(edges(idx(1)),:), 28, 28)', []);
    subplot(1, 3, 2);
    imshow(reshape(train_imgs(edges(idx(2)),:), 28, 28)', []);
    subplot(1, 3, 3);
    imshow(reshape(train_imgs(edges(idx(3)),:), 28, 28)', []);
    fig_path = ['./results/three_fig', num2str(i)];
    saveas(gcf, fig_path, 'epsc');
    
    % weak learners behavior
    a = ones(1, 784) * 128;
    a(alphas(:, 1)) = 127.5 + alphas(:, 3)*127.5;
    figure(5);
    imshow(reshape(a, 28, 28)', []);
    fig_path = ['./results/weak_learner', num2str(i)];
    saveas(gcf, fig_path, 'epsc');
    
    % final classifier
    final_scores = [final_scores, final_test_score];
end


% final classifier and test errors
[~, inds] = max(final_scores, [], 2);
inds = inds - 1;
final_error = 1 - sum(inds == test_labels) * 1.0 / length(test_labels)