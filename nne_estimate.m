
function par = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, varargin)

opt = inputParser;
% addParameter(opt, 'refine', false);
addParameter(opt, 'se', false);
addParameter(opt, 'checks', true);
parse(opt, varargin{:});

warning('off', 'backtrace')

Xp = cast(Xp, nne.numType);
Xa = cast(Xa, nne.numType);
Xc = cast(Xc, nne.numType);

n = size(Xc, 1);
J = size(Xp, 1)/n;

R = 50;

%% pre-estimation checks

if opt.Results.checks
    
    if size(Xp,2) < 2; warning("- Xp has fewer attributes than preset."); end
    if size(Xp,2) > nne.pm; warning("- Xp has more attributes than preset."); end
    if size(Xa,2) > nne.am; warning("- Xa has more attributes than preset."); end
    if size(Xc,2) > nne.cm; warning("- Xc has more attributes than preset."); end
        
    if n < nne.n_range(1); warning("- sample size n is too small: " + n); end
    if J < nne.J_range(1) || J > nne.J_range(2); warning("- number of options J is outside trained range: " + J); end

    if mod(J,1) ~= 0; error('- number of available options is not same across consumers.'); end
    if ~isequal(consumer_idx, reshape(repmat(1:n,J,1), n*J, 1)); error('- consumer_idx is not correct.'); end
    
    [buy_rate, srh_rate, num_srh] = search_stat(Y, consumer_idx);
    
    if buy_rate*n < nne.stat_q.buy(1); warning("- very few made purchases: " + buy_rate*n); end
    if srh_rate*n < nne.stat_q.srh(1); warning("- very few made non-free searches: " + srh_rate*n); end
 
    if buy_rate < nne.stat_q.buy_rate(1); warning("- buy rate is small: " + buy_rate); end
    if buy_rate > nne.stat_q.buy_rate(2); warning("- buy rate is large: " + buy_rate); end
    
    if srh_rate < nne.stat_q.srh_rate(1); warning("- search rate (non-free) is small: " + srh_rate); end
    if srh_rate > nne.stat_q.srh_rate(2); warning("- search rate (non-free) is large: " + srh_rate); end
    
    if num_srh < nne.stat_q.num_srh(1); warning("- avg number of searches is small: " + num_srh); end
    if num_srh > nne.stat_q.num_srh(2); warning("- avg number of searches is large: " + num_srh); end
    
    Zp = zscore(Xp); 
    Za = zscore(Xa);
    Zc = zscore(Xc);
    
    Z = [Zp, Za, Zc(consumer_idx,:)];

    if any( abs(Z) > 3); warning('- X has extreme values.'); end

    f_min = mean( Z==min(Z));
    f_max = mean( Z==max(Z));
    binary = f_min + f_max > 0.99;
    
    if any(   binary & min(f_min, f_max) < 0.03); warning('- X has highly unbalanced dummy variables.'); end
    if any( ~ binary & abs(skewness(Z)) > 2.5); warning('- X has highly skewed attributes.'); end

    Zp_t = accumatrix(consumer_idx, Zp)/J;
    Za_t = accumatrix(consumer_idx, Za)/J;

    if any( std(Zp - Zp_t(consumer_idx,:)) < 0.01); warning('- Xp lacks variations within consumers.'); end
    if any( std(Za - Za_t(consumer_idx,:)) < 0.01); warning('- Xa lacks variations within consumers.'); end
    if any( std(Za - Zp/(Zp'*Zp)*(Zp'*Za)) < 0.01); warning('- Xa essentially enters utility.'); end
    
end

%% estimation

[par, theta] = Estimate(nne, Y, Xp, Xa, Xc, consumer_idx);

%% post-estimation checks

if opt.Results.checks
    
    % dist = abs(theta.val(:,1) - theta.mid);
    % if any( dist > 2.58*theta.std); warning('- estimates are outside the usual parameter range in training.'); end

    % if any( mmt < prior.mmt_q{1} | mmt > prior.mmt_q{2}); warning('- data moments have extreme values.'); end
    
    % theta.val
    diff_pooled = mean(range(theta.val,2).*theta.diff_w);
    if diff_pooled > nne.diff_q(2); warning('- search model may be badly misspecified for data.'); end
    
end

%% bootstrap SE and refinement

if opt.Results.se
    
    VAL = cell(1, R);
    
    parfor r = 1:R
        
        seed = RandStream('threefry');
        seed.Substream = r;
        
        k = randsample(seed, n, n, true);
        i = reshape((k'-1)*J + (1:J)', n*J, 1);
        
        par_boot = Estimate(nne, Y(i,:), Xp(i,:), Xa(i,:), Xc(k,:), consumer_idx);
        
        VAL{r} = par_boot.val;
        
    end

    par.se = std(cell2mat(VAL), [], 2);

end

warning('on', 'backtrace')

end

%% ................................................................................................
%% ................................................................................................

%% Large data splitter

function [par, theta] = Estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

n = size(Xc, 1);
n_max = nne.n_range(2);

if n <= n_max
    
    [par, theta] = Estimator(nne, Y, Xp, Xa, Xc, consumer_idx);
    
else

    T = ceil(n/n_max);
    
    seed = RandStream('twister');
    sector_idx = randi(seed, T, n, 1);
    
    PAR_VAL = cell(1, T);
    THT_VAL = cell(1, T);
    
    for t = 1:T
        
        k = find( sector_idx == t);
        i = ismember(consumer_idx, k);
        
        [par, theta] = Estimator(nne, Y(i,:), Xp(i,:), Xa(i,:), Xc(k,:), consumer_idx(i));
        
        PAR_VAL{t} = par.val;
        THT_VAL{t} = theta.val;
        
    end
    
    par.val   = mean(cell2mat(PAR_VAL), 2);
    theta.val = mean(cat(3, THT_VAL{:}), 3);

end

end


%% Core estimation routine

function [par, theta] = Estimator(nne, Y, Xp, Xa, Xc, consumer_idx)

Zp = zscore(Xp);
Za = zscore(Xa);
Zc = zscore(Xc);

p = size(Xp, 2);
a = size(Xa, 2);
c = size(Xc, 2);

dummy = nne.num2dmy(p,a,c);

mmt = moments(Y, Zp, Za, Zc, consumer_idx, nne);
% mmt_clip = min(prior.mmt_q{2}, max(prior.mmt_q{1}, mmt));

pred_net = predict(nne.net, mmt, exec='cpu');

pred_tree = arrayfun(@(k)predict(nne.tree{k}, mmt), 1:numel(nne.tree), 'uni', false);
pred_tree = cell2mat(pred_tree);

theta = table();

theta.name = nne.name;
theta.val  = [pred_net', pred_tree'];

% theta.mid = net.prior(p,a,c).mid;
% theta.std = net.prior(p,a,c).std;
theta.diff_w = nne.diff_w;

theta = theta(dummy,:);

diff_pooled = mean(range(theta.val,2).*theta.diff_w);
coef = rescale(diff_pooled, "inputmin", nne.diff_q(1), "inputmax", nne.diff_q(2));

theta.val = [theta.val*[1-coef, coef]', theta.val];

alpha0 = theta.val(1                 , 1);
alpha  = theta.val(2       : a+1     , 1);
eta0   = theta.val(a+2               , 1);
eta    = theta.val(a+3     : a+c+2   , 1);
beta   = theta.val(a+c+3   : a+c+p+2 , 1);
% delta  = theta.val(a+c+p+3           , 1);

par = table();

par.name = theta.name;
par.val = [
           alpha0 - sum(alpha./std(Xa)'.*mean(Xa)')
           alpha./std(Xa)'
           eta0   - sum(eta  ./std(Xc)'.*mean(Xc)') + sum(beta ./std(Xp)'.*mean(Xp)')
           eta  ./std(Xc)'
           beta ./std(Xp)'
%            delta
           ];

end


% %% moment trim
% 
% function mmt_trim = TrimMoment(mmt, prior, varargin)
% 
% opt = inputParser;
% addParameter(opt, 'warnings', false);
% parse(opt, varargin{:});
% 
% if ~isrow(mmt)
%     error("The moment input must be a row vector.")
% end
% 
% Q = double(prior.input_Q);
% p = double(prior.input_p);
% 
% i = find(p > 0.0001, 1, 'first');
% j = find(p > 0.9999, 1, 'first');
% 
% p = p(i:j);
% Q = Q(i:j,:);
% 
% % mmt_trim = max(min(Q), min(max(Q), mmt));
% 
% mmt_trim = dq_normalize(mmt, p, Q);
% % mmt = double(mmt);
% 
% end

% i = diag(prior.input_V) > 0;
% d = sum(i);
% 
% V = double(prior.input_V(i,i));
% A = V^-1;
% 
% v = mmt(i);
% 
% u = @(lambda)v/(eye(d) + lambda*A);
% stat = @(u)u*A*u' - chi2inv(0.999,d);
% 
% if stat(u(0)) < 0
%     mmt_trim = mmt;
%     return
% elseif opt.Results.warnings
%     disp("- The input moments are atypical; search model may be badly mis-specified for data.")
%     disp("  pNNE will try its best; take estimates with grain of salt.")
% end
% 
% lambda = fzero(@(lambda)stat(u(lambda)), 0, optimset('display', 'off'));
% 
% u = u(lambda);
% 
% mmt_trim = mmt;
% mmt_trim(i) = u;

% if opt.Results.checks
%     
%     if any( mmt < prior.mmt_q{1} | mmt > prior.mmt_q{4})
%         warning('- Data moments have extreme values.')
%     end
%    
%     load('curve.mat', 'curve')
%     
%     MMT = cell(R, 1);
%     
%     parfor r = 1:R
%         
%         y_simul = search_model(Xp, Xa, Xc, consumer_idx, curve, par.val);
% %         [par_bit, ~, ~      ] = Estimate(net, prior, y_simul, Xp, Xa, Xc, consumer_idx);
%         
% %         y_simul = search_model(Xp, Xa, Xc, consumer_idx, curve, par_bit.val);
%         [~      , ~, mmt_bit] = Estimate(net, prior, y_simul, Xp, Xa, Xc, consumer_idx);
%         
%         MMT{r} = mmt_bit;
%         
%     end
%     
%     um = [-3, 4]*quantile(cell2mat(MMT), [0.1, 0.9]');
%     lm = [ 4,-3]*quantile(cell2mat(MMT), [0.1, 0.9]');
%     
%     find( mmt > um + abs(um)*1e-5 | mmt < lm - abs(lm)*1e-5)
%     
%     if any( mmt > um + abs(um)*1e-5 | mmt < lm - abs(lm)*1e-5)
%         warning(' - Bad model fit.')
%     end
%     
% end
