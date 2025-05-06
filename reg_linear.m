
function [coef, pass] = reg_linear(penalty, X, y)

%{

Estimate a linear model.

%}

n = numel(y);

X = [ones(n,1), X];
w = penalty*[0, ones(1, width(X)-1)];

B = X'*X/n + 2*diag(w);

pass = min(eig(B)) > 1e-9;

coef = (y'*X/n)/B;
