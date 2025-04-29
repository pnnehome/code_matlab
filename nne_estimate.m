
function par = nne_estimate(nne, Y, Xp, Xa, Xc, consumer_idx, varargin)

opt = inputParser;
addParameter(opt, 'se', false);
addParameter(opt, 'se_repeats', 50)
addParameter(opt, 'checks', true);
parse(opt, varargin{:});
opt = opt.Results;              % Apr-7

n = consumer_idx(end);
J = numel(consumer_idx)/n;

if isempty(Xa); Xa = zeros(n*J,0); end
if isempty(Xc); Xc = zeros(n,0); end
if width(Y) > 2; Y = Y(:,1:2); end

[Y, Xp, Xa, Xc] = deal(double(Y), double(Xp), double(Xa), double(Xc));              % Apr-7

%% pre-estimation checks

warning('off', 'backtrace')

if opt.checks

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
    if buy_rate > nne.stat.buy_rate(3); warning("- buy rate is too large."); end
    
    if srh_rate < nne.stat.srh_rate(1); warning("- search rate (non-free) is too small."); end
    if srh_rate > nne.stat.srh_rate(3); warning("- search rate (non-free) is too large."); end
    
    if num_srh < nne.stat.num_srh(1); warning("- avg number of searches is too small."); end
    if num_srh > nne.stat.num_srh(3); warning("- avg number of searches is too large."); end

    % X = [Xp, Xa, Xc(consumer_idx,:)];              % Apr-7

    if any( abs([mean(Xp), mean(Xa), mean(Xc)]) > 1e-5); warning(' - we recommend de-meaning X.'); end              % Apr-7

    bounds = @(T)[0, -1, 2; -2, 1, 0]*prctile(T, [2.5, 50, 97.5]');              % Apr-7

    if any( [max(Xp);-min(Xp)] > bounds(Xp), 'all'); warning('- Xp has extreme values; winsorizing may help.'); end              % Apr-7
    if any( [max(Xa);-min(Xa)] > bounds(Xa), 'all'); warning('- Xa has extreme values; winsorizing may help.'); end              % Apr-7
    if any( [max(Xc);-min(Xc)] > bounds(Xc), 'all'); warning('- Xc has extreme values; winsorizing may help.'); end              % Apr-7
    % if any( max(X) > 2*q(3,:)-q(2,:) | min(X) < 2*q(1,:)-q(2,:)); warning('- X has extreme values; winsorizing may help.'); end

    Zp = zscore(Xp);              % Apr-7
    Za = zscore(Xa);              % Apr-7

    A = sparse(consumer_idx, 1:n*J, 1);              % Apr-7
    Zp_t = A*Zp/J;              % Apr-7
    Za_t = A*Za/J;              % Apr-7

    if any( std(Zp - Zp_t(consumer_idx,:)) < 0.01); warning('- Xp lacks variation within consumers.'); end              % Apr-7
    if any( std(Za - Za_t(consumer_idx,:)) < 0.01); warning('- Xa lacks variation within consumers.'); end              % Apr-7
    if any( std(Za - Zp/(Zp'*Zp)*(Zp'*Za)) < 0.01); warning('- Xa lacks variation independent of Xp.'); end              % Apr-7
    
end

%% shuffle observations              % Apr-7

seed = RandStream('twister', 'seed', 1);
k = randperm(seed, n)';
i = reshape((k'-1)*J + (1:J)', n*J, 1);

Y = Y(i,:);
Xp = Xp(i,:);
Xa = Xa(i,:);
Xc = Xc(k,:);

%% estimation

par = Estimate(nne, Y, Xp, Xa, Xc, consumer_idx);

%% post-estimation checks

if opt.checks

    if     par.Properties.CustomProperties.mmt_sens > 1; warning('- reduced-form patterns are unstable; estimates are likely inaccurate.');
    elseif par.Properties.CustomProperties.mmt_sens > 0.5; warning('- reduced-form patterns are not very stable; estimates may be inaccurate.'); end

    if     par.Properties.CustomProperties.pred_diff > nne.diff.q2; warning('- the data is probably ill-suited for this search model.');
    elseif par.Properties.CustomProperties.pred_diff > nne.diff.q1; warning('- the data might be ill-suited for this search model.'); end

end

warning('on', 'backtrace')

%% bootstrap SE

par = par(:, 'val');

if opt.se             % Apr-7

    vals = cell(1, opt.se_repeats);

    seed = RandStream('twister', 'seed', 2);              % Apr-7
    draws = randi(seed, n, n, opt.se_repeats);              % Apr-7

    parfor r = 1:opt.se_repeats

        k = draws(:, r);              % Apr-7
        i = reshape((k'-1)*J + (1:J)', n*J, 1);              % Apr-7

        par_bt = Estimate(nne, Y(i,:), Xp(i,:), Xa(i,:), Xc(k,:), consumer_idx);
        vals{r} = par_bt.val;              % Apr-7
    end

    par.se = std(cell2mat(vals), [], 2);

end

end

%% SUB-FUNCITONS ............................................................................

%% Large data splitter

function par = Estimate(nne, Y, Xp, Xa, Xc, consumer_idx)

n = height(Xc);
J = height(Xp)/n;

if n <= nne.dim.n(2)
    par = Estimator(nne, Y, Xp, Xa, Xc, consumer_idx);
    return
end

blocks = ceil(n/nne.dim.n(2));              % Apr-7
m = ceil(n/blocks);              % Apr-7
pars = cell(blocks, 1);              % Apr-7

parfor b = 1:blocks              % Apr-7

    k = (b-1)*m + 1 : min(b*m, n);              % Apr-7
    i = ismember(consumer_idx, k);

    par = Estimator(nne, Y(i,:), Xp(i,:), Xa(i,:), Xc(k,:), repelem(1:numel(k), J)' );
    pars{b} = par;
end

pars_mat = cellfun(@table2array, pars, 'uni', false);

par = pars{1};
par{:,:} = mean(cat(3, pars_mat{:}), 3);

end

%% Core estimation routine

function par = Estimator(nne, Y, Xp, Xa, Xc, consumer_idx)

[Zp, mu.p, sigma.p] = zscore(Xp);              % Apr-7
[Za, mu.a, sigma.a] = zscore(Xa);              % Apr-7
[Zc, mu.c, sigma.c] = zscore(Xc);              % Apr-7

p = width(Xp);
a = width(Xa);
c = width(Xc);

i = matches(nne.name, nne.name_active(p, a, c));
par = table('RowNames', nne.name(i));
par = addprop(par, {'mmt_sens', 'pred_diff'}, {'t','t'});

[mmt, sens] = moments(Y, Zp, Za, Zc, consumer_idx, nne);

par.Properties.CustomProperties.mmt_sens = sens;

pred_net = predict(nne.net, mmt);
par.pred_net  = pred_net(i)';

if isfield(nne, 'tree')

    pred_tree = arrayfun(@(k)predict(nne.tree{k}, mmt), 1:numel(nne.tree));
    par.pred_tree = pred_tree(i)';
   
    diff = mean(nne.diff.w(i).*abs(par.pred_net - par.pred_tree));
    par.val = par.pred_net + (par.pred_tree - par.pred_net)*clip((diff - nne.diff.q1)/(nne.diff.q2 - nne.diff.q1), 0, 1);
else
    
    diff = 0;
    par.val = par.pred_net;
end

par.Properties.CustomProperties.pred_diff = diff;

par.val = rescale_estimate(par, mu, sigma);              % Apr-7

end

%% Re-scaling              % Apr-7

function val = rescale_estimate(par, mu, sigma)

alpha0 = par.val("\alpha_0");
alpha  = par.val("\alpha_" + (1:numel(mu.a)));
eta0   = par.val("\eta_0");
eta    = par.val("\eta_" + (1:numel(mu.c)));
beta   = par.val("\beta_" + (1:numel(mu.p)));

val = [
       alpha0 - sum(alpha./sigma.a'.*mu.a')
       alpha./sigma.a'
       eta0   - sum(eta  ./sigma.c'.*mu.c') + sum(beta ./sigma.p'.*mu.p')
       eta  ./sigma.c'
       beta ./sigma.p'
       ];

end

