
function mmt = moments(Y, Zp, Za, Zc, consumer_idx, nne)

%% input processing

% consumer_idx = grp2idx(consumer_idx);

n = height(Zc);
J = height(Zp)/n;

p = width(Zp);
a = width(Za);
c = width(Zc);

ip = [true(1,p), false(1, nne.dim.p(2) - p)];
ia = [true(1,a), false(1, nne.dim.a(2) - a)];
ic = [true(1,c), false(1, nne.dim.c(2) - c)];

ys = Y(:, 1);
yb = Y(:, 2);
% yo = Y(:, 3);

%% construct consumer level

A = sparse(consumer_idx, 1:n*J, 1);

ys_t = A*ys;  % number of searches
yb_t = A*yb;  % whether bought inside goods

Zp_t = A*Zp/J;
Za_t = A*Za/J;

is = logical(ys);
consumer_iis = consumer_idx(is);

Z_s = [Zp,       Za,       Zc(consumer_idx,:), Zp_t(consumer_idx,:), Za_t(consumer_idx,:)];
Z_b = [Zp(is,:), Za(is,:), Zc(consumer_iis,:)];
% Z_o = [Zp(is,:), Za(is,:)];
Z_t = [                    Zc                , Zp_t,                 Za_t];

Z_s_en = [log(ys_t(consumer_idx)), Z_s];
Z_b_en = [log(ys_t(consumer_iis)), Z_b];    % UNCOMMENT IF NO ORDER INFO
% Z_b_en = [log(yo(is)), log(ys_t(consumer_iis)), Z_b];     % COMMENT IF NO ORDER INFO
Z_t_en = [log(ys_t), Z_t];

%% some statistics

mu1 = [mean(ys_t>1), mean(ys_t), mean(log(ys_t)), mean(yb_t), mean(yb(is)), mean(log(ys_t(consumer_iis)))];                
mu2([ip,ia,ic,true]) = [mean(Z_b), nan];

mv1 = [std(ys_t), std(log(ys_t)), std(log(ys_t(consumer_iis)))];
mv2([ip,ia,ic,ip,ia,true]) = [std(Z_b), std(Zp_t), std(Za_t), nan];

%% regressions

[cf1, ~, fl1] = reg_logit( Z_s,    ys,       []);
[cf2, ~, fl2] = reg_logit( Z_s_en, ys,       []);
[cf3, ~, fl3] = reg_logit( Z_b,    yb(is),   consumer_iis);
[cf4, ~, fl4] = reg_logit( Z_b_en, yb(is),   consumer_iis);

[cf5, ~, fl5] = reg_logit( Z_t,    ys_t>1,   []);
[cf6, ~, fl6] = reg_linear(Z_t,    log(ys_t));
[cf7, ~, fl7] = reg_logit( Z_t,    yb_t,     []);
[cf8, ~, fl8] = reg_logit( Z_t_en, yb_t,     []);

if ~ all([fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8])
    mmt = [];
    return
end

% whos mm1 mm2 mm3 mm4 mm5 mm6

%% packaging

mm1([true,     ip,ia,ic,true]) = [cf1(1:p+a+c+1), nan];  % 1-16
mm2([true,true,ip,ia,ic,true]) = [cf2(1:p+a+c+2), nan];  % 17-33
mm3([true,     ip,ia,ic,true]) = [cf3, nan];  % 34-49
mm4([true,true,ip,ia,ic,true]) = [cf4, nan];  % 50-66 

mm5([true,     ic,true]) = [cf5(1:c+1), nan];   % 79-84; 67-72; note: for robustness, intentionally not to use coefs of Zp_t & Za_t.
mm6([true,     ic,true]) = [cf6(1:c+1), nan];   % 85-90; 73-78
mm7([true,     ic,true]) = [cf7(1:c+1), nan];   % 91-96; 79-84
mm8([true,true,ic,true]) = [cf8(1:c+2), nan];  % 97-103; 85-91

% ms1([true,     ip,ia,ic,true]) = [se1(1:p+a+c+1), nan];
% ms2([true,true,ip,ia,ic,true]) = [se2(1:p+a+c+2), nan];
% ms3([true,     ip,ia,ic,true]) = [se3, nan];
% ms4([true,true,ip,ia,ic,true]) = [se4, nan];
% 
% ms5([true,     ic,true]) = [se5(1:c+1), nan];
% ms6([true,     ic,true]) = [se6(1:c+1), nan];
% ms7([true,     ic,true]) = [se7(1:c+1), nan];
% ms8([true,true,ic,true]) = [se8(1:c+2), nan];

%% keys

%% outputs

mmt = [mm1, mm2, mm3, mm4, mm5, mm6, mm7, mm8, mu1, mu2, mv1, mv2, ip, ia, ic, J, sqrt(n)];
mmt = rmmissing(mmt);

% mmt = cast(mmt, numType);


