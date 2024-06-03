
function [coef, se, flag] = reg_linear(X, y)

%{

Estimate a linear model.

%}

penalty = 1e-5;
n = numel(y);

X = [ones(n,1), X];
w = penalty*[0, ones(1, width(X)-1)];

B = X'*X/n + diag(w);

flag = rcond(B) > 1e-6;

coef = (y'*X/n)/B;
coef = double(coef);

V = var(y - X*coef')*B^-1/n;
se = sqrt(diag(V))';
se = double(se);
