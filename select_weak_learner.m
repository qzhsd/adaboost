function [alpha_t, Y_pred] = select_weak_learner(X, Y, U, sample_weights)
% X: N*d, Y: N*1, U: (d*t*2)*3, sample_weights: N*1
% alpha_t: 1*3
% weighted_margins = zeros(size(U, 1), 1);
% for k = 1 : size(U, 1)
%     u_k = U(k, :);
%     x_k = X(:, u_k(1));
%     Y_pred = ones(size(X, 1), 1) * u_k(3);
%     Y_pred(x_k < u_k(2)) = -u_k(3);
%     weighted_margins(k) = sum(Y .* Y_pred .* sample_weights);
% end
% [~, ind] = max(weighted_margins);
% alpha_t = U(ind, :);
% Y_pred = ones(size(X, 1), 1) * alpha_t(3);
% Y_pred(X(:, alpha_t(1)) < alpha_t(2)) = -alpha_t(3);
% end

t = sum(U(:, 1) == 1) / 2;
batch_size = 112;
u_step = t * 2 * batch_size;
weighted_margins = zeros(size(U, 1), 1);

for k = 1 : u_step : size(U, 1)
    % disp(k)
    u_k = U(k : k+u_step-1, :);
    x_k = X(:, u_k(1, 1) : u_k(end, 1));
    
    seg_margins = margins_helper(x_k, u_k, Y, sample_weights);
    weighted_margins(k : k+u_step-1) = seg_margins;
end
[~, ind] = max(weighted_margins);
alpha_t = U(ind, :);
Y_pred = ones(size(X, 1), 1) * alpha_t(3);
Y_pred(X(:, alpha_t(1)) < alpha_t(2)) = -alpha_t(3);
end


function seg_margins = margins_helper(x_k, u_k, Y, sample_weights)
% x_k: N * batchsize, u_k: u_step * 3
[N, d] = size(x_k);
% x_k = repmat(x_k, 2*t, 1);
% x_k = reshape(x_k, N, []);
% thres = repmat(u_k(:, 2), 1, N)';
% y_pred = repmat(u_k(:, 3), 1, N)';
% inds = x_k < thres;
inds = reshape(bsxfun(@lt, reshape(x_k, N, 1, []), reshape(u_k(:, 2), 1, [], d)), N, []); % N * u_step
% y_pred(inds) = -y_pred(inds);
y_pred = bsxfun(@times, 2*inds-1, u_k(:, 3)');
seg_margins = ((Y .* sample_weights)' * y_pred)';
% seg_margins = sum(repmat(Y, 1, size(y_pred, 2)) .* y_pred .* repmat(sample_weights, 1, size(y_pred, 2)), 1)';
end
