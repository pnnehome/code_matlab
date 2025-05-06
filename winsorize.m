
function X = winsorize(X)

% winsorize data at 0.5 and 99.5 percentiles.

qs = prctile(X, [0.5, 99.5]');

lower = qs(1,:);
upper = qs(2,:);

X = max(lower, min(upper, X));

