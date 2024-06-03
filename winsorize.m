
function X = winsorize(X)

% This function winsorize data at 0.5 and 99.5 percentiles.

qs = prctile(X, [0.5, 99.5]');

lower = qs(1,:);
upper = qs(2,:);

X = max(lower, min(upper, X));

%{

This function clips outliers.
An outlier is any point:
(i) above Q(97.5) + [Q(97.5) - Q(50)], or 
(ii) below Q(2.5) - [Q(50) - Q(2.5)].

%}

% qs = prctile(X, [2.5, 50, 97.5]');
% 
% q1 = qs(1,:);
% q2 = qs(2,:);
% q3 = qs(3,:);
% 
% lower = q1 - (q2 - q1);
% upper = q3 + (q3 - q2);
% 
% X = max(lower, min(upper, X));