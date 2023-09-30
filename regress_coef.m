
function coef = regress_coef(X, y, model, varargin)

% [simple logit]
% y = 1 if x*beta' + alfa + error > 0.
% error follows logistic distribution.
% output is [alfa, beta].

% [multinomial logit]
% y(j) = 1 if x(j,:)*beta' + alfa + error > error_outside and is largest among {idx==j}.
% the errors follow extreme-value distribution.
% output is [alfa, beta].

% [ranked logit]
% y(j) = ranking of x(j,:)*beta' + error among {idx==j}, from the smallest to largest.
% the errors follow extreme-value distribution.
% output is [0, beta].


n = size(X,1);
d = size(X,2);

lamda = 1; 

seed = RandStream('mcg16807');

numType = class(X);
tol = 1e-6;
% tol = 1e-5*( numType == "single") + ...
%       1e-6*( numType == "double");

opts = optimoptions('fminunc', 'Disp','off', 'Alg','trust-region', ...
                               'Spec',true, 'HessianFcn','objective', 'Subp','factorization',...
                               'Func',tol);

% opts = optimoptions('fminunc', 'Disp','off', 'Alg','quasi-newton', 'Spec', true);

%%
switch model
    case "logit"
        
    y = cast(y, numType);
    
    s = min(1e4,n);
    
    f = [n-sum(y), sum(y)];
    g = round(min(f, max(s-min(f), s/2)));
    
    i = [datasample(seed, find(~y), g(1), 're', false); datasample(seed, find( y), g(2), 're', false)];
    
    w = [repmat(f(1)/g(1), g(1), 1); repmat(f(2)/g(2), g(2), 1)];
    w = cast(w, numType);

    para0 = zeros(1,1+d);
    
    [para1, ~, ~   ] = fminunc(@(para)loss_binary(X(i,:), y(i), w, para, lamda), para0, opts);
    [para2, ~, flag] = fminunc(@(para)loss_binary(X     , y   , 1, para, lamda), para1, opts);

    coef = para2;

%%
    case "ranked"
    
%     idx = varargin{1};
    [~, ~, idx] = unique(varargin{1});
    L = max(y);
    
    t = accumarray(idx, 1);
    t = t(idx);
    
    data = cell(L-1, 3);

    for l = 1:L-1
        i = t >= l+1 & y <= l+1;
        data(l,:) = {y(i) == l+1, X(i,:), idx(i) + l*max(idx)};
    end
    
    y   = cell2mat(data(:,1));
    X   = cell2mat(data(:,2));
    idx = cell2mat(data(:,3));
    
    [~, ~, idx] = unique(idx);
        
    s = min(5000, max(idx));
    i = ismember(idx, datasample(seed, 1:max(idx), s, 're', false));
    [~, ~, idi] = unique(idx(i));
	
    para0 = zeros(1,d);

    [para1, ~, ~   ] = fminunc(@(para)loss_multinomial(X(i,:), y(i), idi, para, lamda), para0, opts);
    [para2, ~, flag] = fminunc(@(para)loss_multinomial(X     , y   , idx, para, lamda), para1, opts);

    coef = [0, para2];

%%
    case "multi"
        
    [~, ~, idx] = unique(varargin{1});
    
    ins = (1:n)' + idx;
    out = setdiff(1:n+max(idx), ins)';

    X = accumatrix(ins, [ones(n,1,numType), X]);
    
    t = accumarray(idx, y);
    y = accumarray(ins, y) + accumarray(out, 1-t, [n+max(idx), 1]);
    y = logical(y);
    
    idx = accumarray(ins, idx) + accumarray(out, 1:max(idx), [n+max(idx), 1]);
    
    s = min(5000, max(idx));
    i = ismember(idx, datasample(seed, 1:max(idx), s, 're', false));
    [~, ~, idi] = unique(idx(i));
    
    para0 = zeros(1,d+1);
    
    [para1, ~, ~   ] = fminunc(@(para)loss_multinomial(X(i,:), y(i), idi, para, lamda), para0, opts);
    [para2, ~, flag] = fminunc(@(para)loss_multinomial(X     , y   , idx, para, lamda), para1, opts);
    
    coef = para2;
%     coef = [0, para2];
    
%%
    case "ols"
    
    Xbar = mean(X,1);
    ybar = mean(y);
    
    Xu = X - Xbar;
    yu = y - ybar;
    
    beta = (Xu'*Xu + lamda*eye(d))\(Xu'*yu);
    alfa = ybar - Xbar*beta;
    
    coef = double([alfa, beta']);
    flag = 1;

end

if flag <= 0
    error('MLE not successful.')
end

end

%% sub-function: loss for [logit]

function [val, grad, hess] = loss_binary(X, y, w, para, lamda)

d = size(X,2);
alfa = para(1);
beta = para(2:d+1);

v = X*beta' + alfa;
e = exp(-v);

log_p = - (1 - y).*v - log(1+e);
val  = - sum(log_p.*w) + lamda*sum(beta.^2);

diff = (-1 + y + y.*e)./(1+e);
score = [diff, X.*diff];
scorw = score.*w;

grad = - sum(scorw) + 2*lamda*[0, beta];
hess = scorw'*score + 2*lamda*blkdiag(0, eye(d));

val  = double(val);
grad = double(grad);
hess = double(hess);

end

%% sub-function: loss for [multinomial logit]

function [val, grad, hess] = loss_multinomial(X, y, idx, para, lamda)

d = size(X,2);

beta = para;

v = X*beta';
% v = v - mean(v);
m = accumarray(idx, v)./accumarray(idx, 1);
v = v - m(idx);

e = exp(v);
s = accumarray(idx, e);

log_p = accumarray(idx(y), v(y)) - log(s);
val = - sum(log_p) + lamda*sum(beta.^2);

score = accumatrix(idx(y), X(y,:)) - accumatrix(idx, X.*e)./s;

grad = - sum(score) + 2*lamda*beta;
hess = score'*score + 2*lamda*eye(d);

val  = double(val);
grad = double(grad);
hess = double(hess);

end
