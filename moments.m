
function [mmt, sen] = moments(Y, Xp, Xa, Xc, consumer_idx, nne)

%% input processing

% consumer_idx = grp2idx(consumer_idx); 

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
% yo = Y(:, 3);

%%

[mu1, mu2, mu3, mv1, mv2, mu3] = deal([]);
[cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8] = deal([]);
[ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8] = deal([]);
[mm1, mm2, mm3, mm4, mm5, mm6, mm7, mm8] = deal([]);
[ms1, ms2, ms3, ms4, ms5, ms6, ms7, ms8] = deal([]);

%% construct consumer level

A = sparse(consumer_idx, 1:n*J, 1);

ys_t = A*ys;  % number of searches
yb_t = A*yb;  % whether bought inside goods

Xp_t = A*Xp/J;
Xa_t = A*Xa/J;

Xp_s = Xp(ys,:);
Xa_s = Xa(ys,:);
Xc_s = A(:,ys)'*Xc;

% X_s_en = [A'*log(ys_t), X_s];
% X_b_en = [B'*log(ys_t), X_b];
% X_t_en = [log(ys_t), X_t];

%% some statistics

% mu1 = [mean(ys_t>1), mean(ys_t), mean(log(ys_t)), mean(yb_t), mean(yb(is)), mean(B'*log(ys_t))];    
mu1 = [mean(ys_t>1), mean(ys_t), mean(log(ys_t)), mean(yb_t), mean(yb(ys))];
mu2([ip,ia,ic,true]) = [mean(Xp_s), mean(Xa_s), mean(Xc_s), nan];

% mv1 = [std(ys_t), std(log(ys_t)), std(B'*log(ys_t))];
mv1 = [std(ys_t), std(log(ys_t))];
mv2([ip,ia,ic,true]) = [std(Xp_s), std(Xa_s), std(Xc_s), nan];
mv3([ip,ia,   true]) = [std(Xp_t), std(Xa_t), nan];

%% regressions

Xp_t(:, std(Xp_t) < 1e-4) = [];
Xa_t(:, std(Xa_t) < 1e-4) = [];

X_o = [Xp,   Xa,   A'*Xc,  A'*Xp_t,  A'*Xa_t];
X_s = [Xp_s, Xa_s, Xc_s];
X_t = [            Xc,     Xp_t,     Xa_t];

% [cf6, se6, fl6] = reg_logit( X_s_en, ys,       [], penalty);
% [cf7, se7, fl7] = reg_logit( X_b_en, yb(is),   consumer_idx(is), penalty);
% [cf8, se8, fl8] = reg_logit( X_t_en, yb_t,     [], penalty);

% lamda = 1e-5;

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

% if max(abs(sen)) > 1
%     mmt = [];
%     return
% end

%% packaging

% whos mm1 mm2 mm3 mm4 mm5 mm6 mm7 mm8 mu1 mu2 mv1 mv2 ip ia ic J n

%% outputs

mmt = [mm1(:)', mm2(:)', mm3(:)', mm4(:)', mm5(:)', mm6(:)', mm7(:)', mm8(:)', ...
       ms1(:)', ms2(:)', ms3(:)', ms4(:)', ms5(:)', ms6(:)', ms7(:)', ms8(:)', ... 
       mu1, mu2, mu3, mv1, mv2, mv3, ...
       ip, ia, ic, J, sqrt(n)];

mmt = rmmissing(mmt);


