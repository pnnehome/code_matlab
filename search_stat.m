
function [buy_rate, srh_rate, num_srh, pass] = search_stat(Y, consumer_idx)

ys = Y(:,1)==1;
yb = Y(:,2)==1;

ys_t = accumarray(consumer_idx, ys);  % number of searches
yb_t = accumarray(consumer_idx, yb);  % whether bought inside goods

if max(yb_t) > 1; error("- someone bought more than one options."); end
if min(ys_t) < 1; error("- someone did not make the free search."); end
if any(yb & ~ys); error("- someone bought an option without searching it."); end

buy_rate = mean(yb_t);
srh_rate = mean(ys_t > 1);
num_srh = mean(ys_t);

n = numel(ys_t);
J = numel(ys)/n;
   
pass = buy_rate > 20/n  && ... 
       buy_rate > 0.005 && ... 
       buy_rate < 0.99  && ...
       srh_rate > 50/n  && ...
       srh_rate > 0.01  && ...
       num_srh < J-1;