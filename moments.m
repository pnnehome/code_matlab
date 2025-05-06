
function [mmt, sen] = moments(Y, Xp, Xa, Xc, consumer_idx, nne)

%% input processing

n = height(Xc);
J = height(Xp)/n;

p = width(Xp);
a = width(Xa);
c = width(Xc);

ip = [true(1,p), false(1, nne.dim.p(2) - p)];
ia = [true(1,a), false(1, nne.dim.a(2) - a)];
ic = [true(1,c), false(1, nne.dim.c(2) - c)];

lamda = nne.reg_lamda;

ys = logical(Y(:, 1));
yb = logical(Y(:, 2));

%%

[mu1, mu2, mu3, mv1, mv2, mu3] = deal([]);
[cf1, cf2, cf3, cf4, cf5] = deal([]);
[ps1, ps2, ps3, ps4, ps5] = deal([]);
[mm1, mm2, mm3, mm4, mm5] = deal([]);

%% construct consumer level

A = sparse(consumer_idx, 1:n*J, 1);

ys_t = A*ys;  % number of searches
yb_t = A*yb;  % whether bought inside goods

Xp_t = A*Xp/J;
Xa_t = A*Xa/J;

Xp_s = Xp(ys,:);
Xa_s = Xa(ys,:);
Xc_s = A(:,ys)'*Xc;

%% some statistics

mu1 = [mean(ys_t>1), mean(ys_t), mean(log(ys_t)), mean(yb_t), mean(yb(ys))];
mu2([ip,ia,ic,true]) = [mean(Xp_s), mean(Xa_s), mean(Xc_s), nan];

mv1 = [std(ys_t), std(log(ys_t))];
mv2([ip,ia,ic,true]) = [std(Xp_s), std(Xa_s), std(Xc_s), nan];
mv3([ip,ia,   true]) = [std(Xp_t), std(Xa_t), nan];

%% regressions

Xp_t(:, std(Xp_t) < 1e-4) = [];
Xa_t(:, std(Xa_t) < 1e-4) = [];

X_o = [Xp,   Xa,   A'*Xc,  A'*Xp_t,  A'*Xa_t];
X_s = [Xp_s, Xa_s, Xc_s];
X_t = [            Xc,     Xp_t,     Xa_t];


for k = 1:length(lamda)
    
    [cf1, ps1] = reg_logit (lamda(k), X_o,  ys,       []);
    [cf2, ps2] = reg_logit (lamda(k), X_s,  yb(ys),   consumer_idx(ys));
    [cf3, ps3] = reg_logit (lamda(k), X_t,  ys_t>1,   []);
    [cf4, ps4] = reg_linear(lamda(k), X_t,  log(ys_t));
    [cf5, ps5] = reg_logit (lamda(k), X_t,  yb_t,     []);

    if ~ all([ps1, ps2, ps3, ps4, ps5])
        mmt = [];
        sen = nan;
        return
    end

    mm1(k, [true,ip,ia,ic,true]) = [cf1(1:p+a+c+1), nan];
    mm2(k, [true,ip,ia,ic,true]) = [cf2, nan];
    mm3(k, [true,ic,true]) = [cf3(1:c+1), nan];
    mm4(k, [true,ic,true]) = [cf4(1:c+1), nan];
    mm5(k, [true,ic,true]) = [cf5(1:c+1), nan];

end

sen = max(abs(diff([mm1, mm2, mm3, mm4, mm5])));


%% outputs

mmt = [mm1(:)', mm2(:)', mm3(:)', mm4(:)', mm5(:)', ...
       mu1, mu2, mu3, mv1, mv2, mv3, ...
       ip, ia, ic, J, sqrt(n)];

mmt = rmmissing(mmt);


