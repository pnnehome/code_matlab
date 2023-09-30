
function [mmt, key] = moments(Y, Zp, Za, Zc, consumer_idx, nne)

%% input processing

[~, ~, consumer_idx] = unique(consumer_idx);

numType = class(Zp);

n = size(Zc, 1);
J = size(Zp, 1)/n;

pm = nne.pm;
am = nne.am;
cm = nne.cm;

p = size(Zp, 2);
a = size(Za, 2);
c = size(Zc, 2);

ip = [true(1,p), false(1,pm-p)];
ia = [true(1,a), false(1,am-a)];
ic = [true(1,c), false(1,cm-c)];

ys = Y(:, 1);
yb = Y(:, 2);
% yo = Y(:, 3);

%% construct consumer level

ys_t = accumarray(consumer_idx, ys);  % number of searches
yb_t = accumarray(consumer_idx, yb);  % whether bought inside goods

Zp_t = accumatrix(consumer_idx, Zp)/J;
Za_t = accumatrix(consumer_idx, Za)/J;

is = logical(ys);
consumer_iis = consumer_idx(is);

Z   = [Zp,       Za,       Zc(consumer_idx,:), Zp_t(consumer_idx,:), Za_t(consumer_idx,:)];
Z_b = [Zp(is,:), Za(is,:), Zc(consumer_iis,:)];
% Z_o = [Zp(is,:), Za(is,:)];
Z_t = [                    Zc                , Zp_t,                 Za_t];

Z_en   = [log(ys_t(consumer_idx)), Z];
Z_b_en = [log(ys_t(consumer_iis)), Z_b];      % UNCOMMENT IF NO ORDER INFO
% Z_b_en = [log(yo(is)), log(ys_t(consumer_iis)), Z_b]; % COMMENT IF NO ORDER INFO
Z_t_en = [log(ys_t), Z_t];

%% regressions

cf1 = regress_coef(Z,      ys,     'logit');
cf2 = regress_coef(Z_en,   ys,     'logit');
cf3 = regress_coef(Z_b,    yb(is), 'multi',  consumer_iis);
cf4 = regress_coef(Z_b_en, yb(is), 'multi',  consumer_iis);
% cf5 = regress_coef(Z_o,    yo(is), 'ranked', consumer_iis); % COMMENT IF NO ORDER INFO

cf6 = regress_coef(Z_t,    ys_t>1,    'logit');
cf7 = regress_coef(Z_t,    log(ys_t), 'ols');
cf8 = regress_coef(Z_t,    yb_t,      'logit');
cf9 = regress_coef(Z_t_en, yb_t,      'logit');

%% packaging

mm1 = zeros(1,pm+am+cm+1); mm1([true(1,1),ip,ia,ic]) = cf1(1:p+a+c+1);  % 1-16
mm2 = zeros(1,pm+am+cm+2); mm2([true(1,2),ip,ia,ic]) = cf2(1:p+a+c+2);  % 17-33
mm3 = zeros(1,pm+am+cm+1); mm3([true(1,1),ip,ia,ic]) = cf3(1:p+a+c+1);  % 34-49
mm4 = zeros(1,pm+am+cm+2); mm4([true(1,2),ip,ia,ic]) = cf4(1:p+a+c+2);  % 50-66 % UNCOMMENT IF NO ORDER INFO
% mm4 = zeros(1,pm+am+cm+3); mm4([true(1,3),ip,ia,ic]) = cf4(1:p+a+c+3);  % 50-67 % COMMENT IF NO ORDER INFO
% mm5 = zeros(1,pm+am+1   ); mm5([true(1,1),ip,ia   ]) = cf5(1:p+a+1  );  % 68-78 % COMMENT IF NO ORDER INFO

mm6 = zeros(1,cm+1); mm6([true,ic]) = cf6(1:c+1);   % 79-84; 67-72; note: for robustness, intentionally not to use coefs of Zp_t & Za_t.
mm7 = zeros(1,cm+1); mm7([true,ic]) = cf7(1:c+1);   % 85-90; 73-78
mm8 = zeros(1,cm+1); mm8([true,ic]) = cf8(1:c+1);   % 91-96; 79-84
mm9 = zeros(1,cm+2); mm9([true(1,2),ic]) = cf9(1:c+2);  % 97-103; 85-91

% whos mm1 mm2 mm3 mm4 mm5 mm6 mm7 mm8 mm9

stat = [mean(ys_t>1), mean(ys_t), mean(log(ys_t)), mean(yb_t)]; % 104-107; 92-95

%% keys

kp = [repmat('p',1,p), repmat('n',1,pm-p)];
ka = [repmat('a',1,a), repmat('n',1,am-a)];
kc = [repmat('c',1,c), repmat('n',1,cm-c)];

ke1 = ['o',  kp, ka, kc];
ke2 = ['oo', kp, ka, kc];
ke3 = ['o',  kp, ka, kc];
ke4 = ['oo', kp, ka, kc];
% ke5 = ['o',  kp, ka];
ke6 = ['o',  kc];
ke7 = ['o',  kc];
ke8 = ['o',  kc];
ke9 = ['oo', kc];
ke10 = repmat('o', 1, 4);
ke11 = repmat('o', 1, pm+am+cm+2);

key = [ke1, ke2, ke3, ke4, ke6, ke7, ke8, ke9, ke10, ke11];

%% outputs

mmt = [mm1, mm2, mm3, mm4, mm6, mm7, mm8, mm9, stat, ip, ia, ic, J, sqrt(n)]; % UNCOMMENT IF NO ORDER INFO
% mmt = [mm1, mm2, mm3, mm4, mm5, mm6, mm7, mm8, mm9, stat, ip, ia, ic, J, sqrt(n)]; % COMMENT IF NO ORDER INFO

mmt = cast(mmt, numType);


