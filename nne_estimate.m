
function par = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, varargin)

opt = inputParser;
addParameter(opt, 'se', false);
addParameter(opt, 'se_repeats', 50)
addParameter(opt, 'checks', true);
parse(opt, varargin{:});

n = consumer_idx(end);
J = numel(consumer_idx)/n;

if isempty(Xa); Xa = zeros(n*J,0); end
if isempty(Xc); Xc = zeros(n,0); end

Xp = cast(Xp, nne.numType);
Xa = cast(Xa, nne.numType);
Xc = cast(Xc, nne.numType);
Y  = cast(Y,  nne.numType);

%% pre-estimation checks

warning('off', 'backtrace')

if opt.Results.checks

    [~, buy_rate, srh_rate, num_srh, buy_n, srh_n] = data_checks(Y, Xp, Xa, Xc, consumer_idx);

    if buy_rate == 0; error('- there are no purchases.'); end

    if width(Xp) < nne.dim.p(1); error("- number of attributes in Xp must be at least " + nne.dim.p(1)); end
    if width(Xp) > nne.dim.p(2); error("- number of attributes in Xp must be at most "  + nne.dim.p(2)); end
    if width(Xa) > nne.dim.a(2); error("- number of attributes in Xa must be at most "  + nne.dim.a(2)); end
    if width(Xc) > nne.dim.c(2); error("- number of attributes in Xc must be at most "  + nne.dim.c(2)); end
    
    if n < nne.dim.n(1); warning("- sample size n must be at least: " + nne.dim.n(1)); end
    if J < nne.dim.J(1); warning("- number of options J must be at least: " + nne.dim.J(1)); end
    if J > nne.dim.J(2); warning("- number of options J must be at most: "  + nne.dim.J(2)); end

    if buy_n < nne.stat.buy_n(1); warning("- very few consumers made purchases."); end
    if srh_n < nne.stat.srh_n(1); warning("- very few consumers non-free searches."); end
 
    if buy_rate < nne.stat.buy_rate(1); warning("- buy rate is too small."); end
    if buy_rate > nne.stat.buy_rate(2); warning("- buy rate is too large."); end
    
    if srh_rate < nne.stat.srh_rate(1); warning("- search rate (non-free) is too small."); end
    if srh_rate > nne.stat.srh_rate(2); warning("- search rate (non-free) is too large."); end
    
    if num_srh < nne.stat.num_srh(1); warning("- avg number of searches is too small."); end
    if num_srh > nne.stat.num_srh(2); warning("- avg number of searches is too large."); end

    [Zp, Za, Zc] = deal(zscore(Xp), zscore(Xa), zscore(Xc));
    Z = [Zp, Za, Zc(consumer_idx,:)];

    q = prctile(Z, [2.5, 50, 97.5]');

    if any( max(Z) > 2*q(3,:)-q(2,:) | min(Z) < 2*q(1,:)-q(2,:)); warning('- X has extreme values; winsorizing may help.'); end
    if any( abs(skewness(Z)) > 2 & abs(mode(round(Z,1))) > 0.5); warning('- X has highly skewed attributes.'); end

    A = sparse(consumer_idx, 1:n*J, 1);
    Zp_t = A*Zp/J;
    Za_t = A*Za/J;

    if any( std(Zp - Zp_t(consumer_idx,:)) < 0.01); warning('- Xp lacks variation within consumers.'); end
    if any( std(Za - Za_t(consumer_idx,:)) < 0.01); warning('- Xa lacks variation within consumers.'); end
    if any( std(Za - Zp/(Zp'*Zp)*(Zp'*Za)) < 0.01); warning('- Xa lacks variation independent of Xp.'); end
    
end

%% estimation

par = Estimate(nne, Y, Xp, Xa, Xc, consumer_idx);

%% post-estimation checks

if opt.Results.checks 
    if isfield(nne, 'tree') && mean(par.pred_diff) > nne.diff.q2; warning('- the data may be ill-suited for this search model.'); end
end

warning('on', 'backtrace')

%% bootstrap SE

par = par(:, 'val');
if ~ opt.Results.se
    return
end

vals = cell(1, opt.Results.se_repeats);
idx_cabinet = arrayfun(@(i)(i-1)*J+1:i*J, 1:n, 'uni', false);

parfor r = 1:opt.Results.se_repeats

    seed = RandStream('threefry');
    seed.Substream = r;

    k = randsample(seed, n, n, true);
    i = cell2mat(idx_cabinet(k))';

    par_bt = Estimate(nne, Y(i,:), Xp(i,:), Xa(i,:), Xc(k,:), consumer_idx);
    vals{r} = par_bt.val;
end

par.se = std(cell2mat(vals), [], 2);

end

%% ................................................................................................
%% ................................................................................................

%% Large data splitter

function par = Estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

n = height(Xc);

if n <= nne.dim.n(2)
    par = Estimator(nne, Y, Xp, Xa, Xc, consumer_idx);
    return
end

blocks = ceil(n/nne.dim.n(2));

seed = RandStream('twister');
block_idx = ceil(randperm(seed, n)'/n*blocks);

pars = cell(blocks, 2);

parfor b = 1:blocks

    k = find( block_idx == b);
    i = ismember(consumer_idx, k);

    par = Estimator(nne, Y(i,:), Xp(i,:), Xa(i,:), Xc(k,:), grp2idx(consumer_idx(i)) );
    pars{b} = par;
end

pars_mat = cellfun(@table2array, pars, 'uni', false);

par = pars{1};
par{:,:} = mean(cat(3, pars_mat{:}), 3);

end

%% Core estimation routine

function par = Estimator(nne, Y, Xp, Xa, Xc, consumer_idx)

[Zp, mu_p, sigma_p] = zscore(Xp);
[Za, mu_a, sigma_a] = zscore(Xa);
[Zc, mu_c, sigma_c] = zscore(Xc);

p = size(Xp, 2);
a = size(Xa, 2);
c = size(Xc, 2);

i = matches(nne.name, nne.name_active(p, a, c));
par = table('RowNames', nne.name(i));

mmt = moments(Y, Zp, Za, Zc, consumer_idx, nne);

pred_net = predict(nne.net, mmt);
par.pred_net  = pred_net(i)';

if isfield(nne, 'tree')

    pred_tree = arrayfun(@(k)predict(nne.tree{k}, mmt), 1:numel(nne.tree));
    par.pred_tree = pred_tree(i)';
    par.pred_diff = nne.diff.w(i).*abs(pred_net(i)' - pred_tree(i)');

    diff = mean(par.pred_diff);
    w = (diff - nne.diff.q1)/(nne.diff.q2 - nne.diff.q1);

    par.pred = par.pred_net + clip(w,0,1)*(par.pred_tree - par.pred_net);
else
    
    par.pred = par.pred_net;
end

alpha0 = par.pred("\alpha_0");
alpha  = par.pred("\alpha_" + (1:a));
eta0   = par.pred("\eta_0");
eta    = par.pred("\eta_" + (1:c));
beta   = par.pred("\beta_" + (1:p));

par.val = [
           alpha0 - sum(alpha./sigma_a'.*mu_a')
           alpha./sigma_a'
           eta0   - sum(eta  ./sigma_c'.*mu_c') + sum(beta ./sigma_p'.*mu_p')
           eta  ./sigma_c'
           beta ./sigma_p'
           ];

end

