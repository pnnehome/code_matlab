
function [coef, pass] = reg_logit(penalty, X, y, consumer_idx)

%{

Estimate a multinomial logit model.

y(j) = 1 if beta_0 + x(j,:)*beta' + error is larger than outside utility 
and is largest among {k: consumer_idx(k)=consumer_idx(j)}. The error is 
extreme value distributed. The output coef = [beta_0, beta].

To run a simple logit, set consumer_idx = []. It should give the same
result as setting  consumer_idx=(1:n)', but is faster.

flag = 0 indicates that the regression has failed.

%}

m = 1e4;
% penalty = 1e-5;
% seed = RandStream('mcg16807', 'seed', 0);

y = logical(y);
X = [ones(size(y)), X];
coef = zeros(1, width(X));

opt = optimoptions('fminunc', 'disp','off', 'alg','trust-region', 'spec',true, 'hessianf','objective', 'subp','factorization');

if isempty(consumer_idx)    % simple logit

    n = numel(y);

    if n > m

        n1 = nnz(y);
        n0 = n - n1;
        m1 = min(n1, max(m - n0, m/2));
        m0 = m - m1;

        i = [find(y, m1); find(~y, m0)];

        % i = [datasample(seed, find(y), m1, re=false); datasample(seed, find(~y), m0, re=false)];

        coef = fminunc(@(coef)loss_binary(X(i,:), y(i), penalty, coef), coef, opt);
        coef(1) = coef(1) - log(n0/n1*m1/m0);
    end

    [coef,~,flag,~,~,hess] = fminunc(@(coef)loss_binary(X, y, penalty, coef), coef, opt);

else  % multinomial logit

    n = max(consumer_idx);

    if n > m

        t = accumarray(consumer_idx, y);
        m1 = max(1, floor(mean(t)*m));
        m0 = m - m1;

        k = [find(t, m1); find(~t, m0)];

        % k = [datasample(seed, find(t), m1, re=false); datasample(seed, find(~t), m0, re=false)];
        i = ismember(consumer_idx, k);
        
        coef = fminunc (@(coef)loss_multi(X(i,:), y(i), grp2idx(consumer_idx(i)), penalty, coef), coef, opt);
    end

    [coef,~,flag,~,~,hess] = fminunc(@(coef)loss_multi(X, y, consumer_idx, penalty, coef), coef, opt);

end

pass = flag > 0 && min(eig(hess)) > 1e-9;

% sens = - 2*[0, coef(2:end)]/hess;

% V = hess^-1/n;
% se = sqrt(diag(V))';

end

%% sub-function: loss function

function [val, grad, hess] = loss_binary(X, y, penalty, coef)

n = numel(y);
w = penalty*[0, ones(1, width(X)-1)];

v = X*coef';
% e = exp(v);
e = exp(-v);
s = e + 1;

% val = (1/n)*( - sum(v(y)) + sum(log(s)) + sum(w.*coef.^2));
val = (1/n)*( - sum(v(y)) + sum(v) + sum(log(s))) + sum(w.*coef.^2);

if nargout > 1

    % Q = X.*(e./s);
    Q = X./s;
    grad = (1/n)*( - sum(X(y,:)) + sum(Q)) + 2*w.*coef;
    hess = (1/n)*((X - Q)'*Q) + 2*diag(w);
    hess = (hess + hess')/2;
end
end

%% sub-function: loss function

function [val, grad, hess] = loss_multi(X, y, idx, penalty, coef)

n = max(idx);
w = penalty*[0, ones(1, width(X)-1)];

v = X*coef';
e = exp(v);
s = accumarray(idx, e) + 1;

val = (1/n)*( - sum(v(y)) + sum(log(s))) + sum(w.*coef.^2);

if nargout > 1

    Q = X.*(e./s(idx));
    grad = (1/n)*( - sum(X(y,:)) + sum(Q)) + 2*w.*coef;

    H = sparse(idx, 1:numel(idx), 1)*Q;
    % H = accumatrix(idx, Q);
    hess = (1/n)*( X'*Q - H'*H) + 2*diag(w);
    hess = (hess + hess')/2;
end
end

% %% sub-function: quadratic optimization
% 
% function [para, hess, flag] = minimize(func, para) 
% 
% max_iter = 100;
% tol_step = 1e-7;
% tol_func = 1e-7;
% 
% tau = 1e-5;
% min_tau = 1e-7;
% 
% flag = 0;
% 
% for iter = 1:max_iter
% 
%     [val, grad, hess] = func(para);
%     % [val, grad, hess] = deal(double(val), double(grad), double(hess));
% 
%     % disp([val, tau, para])
% 
%     if ~ allfinite([val, grad, hess(:)'])
%         return
%     end
% 
%     while true 
% 
%         step = - grad/(hess + tau*eye(height(hess)));
%         drop = val - func(para + step);
% 
%         if max(abs(step)) < tol_step
%             flag = 1;
%             return
%         elseif drop > tol_func
%             para = para + step;
%             tau = max(tau/10, min_tau);
%             break
%         else
%             tau = tau*10;
%         end
% 
%     end
% 
% end
% 
% end
