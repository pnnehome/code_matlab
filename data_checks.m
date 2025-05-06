
function [pass, buy_rate, srh_rate, num_srh, buy_n, srh_n] = data_checks(Y, Xp, Xa, Xc, consumer_idx)

n = consumer_idx(end);
count = sparse(consumer_idx, 1, 1);
J = full(count(1));

if any( count == 0); error('- consumer_idx should have consecutive numbers.'); end
if range(count) > 0; error('- current version requires the number of options to be constant across consumers.'); end
if ~ isequal(consumer_idx, repelem((1:n)', J)); error('- consumer_idx is not correct.'); end

if isempty(Xa); Xa = zeros(n*J, 0); end
if isempty(Xc); Xc = zeros(n, 0); end

if width(Y) ~= 2; error('- number of columns of Y should be 2.'); end
if height(Y) ~= n*J; error('- number of rows in Y should equal the length of consumer_idx'); end
if ~ isequal(Y, logical(Y)); error('- Y must be binary.'); end

if height(Xp) ~= n*J; error('- number of rows in Xp should equal the length of consumer_idx'); end
if height(Xa) ~= n*J; error('- number of rows in Xp should equal the length of consumer_idx'); end
if height(Xc) ~= n;   error('- number of rows in Xc should equal the number of consumers n.'); end

if ~ ( allfinite(Xp) && allfinite(Xa) && allfinite(Xc)); error(' - X has missing or non-finite values.'); end
if any( [range(Xp), range(Xa), range(Xc)] == 0); error(' - X has attributes without variations.'); end

ys = Y(:,1);
yb = Y(:,2);

ys_t = accumarray(consumer_idx, ys);  % number of searches
yb_t = accumarray(consumer_idx, yb);  % whether bought inside goods

if max(yb_t) > 1; error("- someone bought more than one options."); end
if min(ys_t) < 1; error("- someone did not make the free search."); end
if any(yb & ~ys); error("- someone bought an option without searching it."); end

buy_rate = mean(yb_t);
srh_rate = mean(ys_t > 1);
num_srh = mean(ys_t);
buy_n = buy_rate*n;
srh_n = srh_rate*n;

pass = buy_n    > 25  && ... 
       buy_rate > 0.005 && ... 
       buy_rate < 0.99  && ...
       srh_n    > 25  && ...
       srh_rate > 0.005  && ...
       num_srh  < J-1;
