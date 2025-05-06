
function Y = search_model(par, curve, Xp, Xa, Xc, consumer_idx)

p = width(Xp);
a = width(Xa);
c = width(Xc);

if numel(par) ~= p+a+c+2
    error("incorrect number of parameters.")
end

n = height(Xc);     % number of consumers
J = height(Xp)/n;   % number of search options

par = par(:)';

alpha0 = par(1                 );
alpha  = par(2       : a+1     );
eta0   = par(a+2               );
eta    = par(a+3     : a+c+2   );
beta   = par(a+c+3   : a+c+p+2 );
delta  = 0;

o = eta0 + Xc*eta' + randn(n, 1);
v = Xp*beta' - o(consumer_idx) + randn(n*J, 1)*exp(delta);
u = v + randn(n*J, 1);

r = v + interpolate(alpha0 + Xa*alpha', curve.log_cost, curve.utility);

[~, sort_idx] = sortrows([consumer_idx, -r]);

u = u(sort_idx);
r = r(sort_idx);

Y = nan(n*J, 2);

for i = 1:n
    
    k = (i-1)*J+1:i*J;
    
    r_i = r(k);
    u_i = u(k);
    
    searched = cummax([0; u_i(1:J-1)]) <= r_i;
    searched(1) = 1;
    
    u_i( ~ searched) = -inf;
    
    [~, j] = max(u_i);
    bought = false(J,1);
    bought(j) = max(u_i) > 0;
    
    Y(k,1) = searched;
    Y(k,2) = bought;
    
end

Y(sort_idx,:) = Y;

end

% sub-function: interpolation

function output = interpolate(query, x, v)

    % x needs to be sorted from smallest to largest
    % x may NOT contain duplicates.
    % query must be in [x(1), x(end)].
    
    [~,j] = max( query <= x', [], 2);
    j = max(2,j);
    
    xl = x(j-1);
    xr = x(j);
    
    vl = v(j-1);
    vr = v(j);
    
    w = (xr - query)./(xr - xl);
    
    output = w.*vl + (1-w).*vr;

end
