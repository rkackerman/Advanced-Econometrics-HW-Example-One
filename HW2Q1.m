clear;
clc;


% Robert Ackerman
% Homework 2
% Problem 1
% October 15, 2013

% Original Paper: R. Clarida, J. Gali, & M. Gertler (2000).
% Monetary Policy Rules and Macroeconomic Stability: Evidence
% and Some Theory. Quarterly Journal of Economics, 115, 147-180.

%% Step 1: Preliminary Settings
% choose subsamples
subsample = 1;  % 1 for Pre-Volcker (60Q1-79Q2; 1-78; #78)
                % 2 for Volcker-Greenspan (79Q3-96Q4; 79-148; #70)
                % 3 for Post-CGG (97Q1-2012Q2; 149-210; #62)
% choose horizons                
k = 4;   % horizon for inflation (must be 1 or 4)
q = 1;   % horizon for output gap (must be 1 or 2)

% choose initial parameter values 
thetainit1 = [0.83; 0.27; 0.68; 4.24];  % Pre-Volcker
thetainit2 = [2.15; 0.93; 0.79; 3.58];  % Volcker-Greenspan 
thetainit3 = [2.15; 0.93; 0.79; 3.58];  % Post-CGG

% # of lags of instruments
nlag = 4;

% set fminunc options for NLS 
optionsNLS = optimset('Display','iter','GradObj','on','LargeScale','on');
% set fminunc options for first-stage GMM
optionsGMM1 = optimset('Display','iter','LargeScale','off');
% set fminunc options for second-stage GMM
optionsGMM2 = optimset('Display','iter', 'GradObj', 'on','LargeScale','on');

% Preparation for 95% confidence region
betavec1  = (-2:0.1:6)';    % grid for beta in subsample 1
gammavec1 = (-2:0.1:6)';   % grid for gamma in subsample 1
betavec2  = (-2:0.1:6)';    % grid for beta in subsample 2
gammavec2 = (-2:0.1:6)';   % grid for gamma in subsample 2
betavec3  = (-2:0.1:6)';    % grid for beta in subsample 3
gammavec3 = (-2:0.1:6)';   % grid for gamma in subsample 3

%% Step 2: Data Management
load HW2Q1b.txt;   % 210-by-9 data matrix
data = HW2Q1b;

% HW2Q1b contains 
% [r, r(-1), pi1, pi4, X1, X2, Commodity price growth, M2 growth, Spread]
% from 1960Q1 - 2012Q2
FF = data(:,1);     FF_lag = data(:,2);  pi1 = data(:,3);
pi4 = data(:,4);    X1 = data(:,5);      X2 = data(:,6);
COM = data(:,7);    M2 = data(:,8);      SPR = data(:,9);

% choose inflation measure (k must be 1 or 4)
if k == 1
    inflation = pi1;
elseif k == 4
    inflation = pi4;
else
    error('Invalid k');
end;

% choose output gap measure (q must be 1 or 2)
if q == 1
    X = X1;
elseif q == 2
    X = X2;
else
    error('Invalid q');
end;

% Based on CGG's theory, we do not have to use Newey-West HAC estimator when k = q = 1.
% We do have to use it otherwise.
if (k == 1) && (q == 1)
    lambda = 0;     % no HAC
else
    lambda = 'NW' ;  % HAC with Newey and West's (1994) automatic lag selection.
end;

T = size(FF, 1);   % entire sample size

% collect all variables in model
data_in_model = [FF, inflation, X, FF_lag];

%% Step 3: Nonlinear Least Squares (NLS)
% choose subsample automatically
if subsample == 1   % Pre-Volcker (60Q1-79Q2; 1-78)
    ssind = 1:78;
elseif subsample == 2   % Volcker-Greenspan (79Q3-96Q4; 79-148)
    ssind = 79:148;
elseif subsample == 3   % Post-CGG (97Q1-2012Q2; 149-210)
    ssind = 163:T;
end;

data_NLS = data_in_model(ssind,:);   % instruments not needed for NLS

% choose initial parameter values automatically
if subsample == 1
    thetainit = thetainit1;
elseif subsample == 2   
    thetainit = thetainit2;
elseif subsample == 3
    thetainit = thetainit3;
end;

% get NLS estimator
theta_NLS = fminunc('NLSobj', thetainit, optionsNLS, data_NLS);

% get asymptotic covariance matrix for theta_NLS
[~, ~, ~, cov_NLS] = NLSobj(theta_NLS, data_NLS);
se_NLS = sqrt(diag(cov_NLS));

% save results
table_NLS = [theta_NLS, se_NLS];

%% Step 4: Two-Stage Generalized Method of Moments (GMM)
r = 6 * nlag;         % # of instruments
nobs = T - nlag;      % effective sample size

% construct instruments (nobs x r)
% First nlag columns have lags of FF_lag, second nlag columns have lags of
% pi1, ..., last nlag columns have lags of SPR.
Z = zeros(nobs, r);
for i = 1:nlag
     first = nlag + 1 - i;
     last = T - i;
     tind = first:last;
     lagind = i:nlag:(5 * nlag + i);
     Z(:, lagind) = [FF_lag(tind), pi1(tind), X1(tind), COM(tind), M2(tind), SPR(tind)];
end;

% collect variables in model
% choose subsample automatically
% discard the first nlag observations
data_GMM = data_in_model((nlag + 1):T, :);
if subsample == 1
    ssind = 1:(78-nlag);
elseif subsample == 2
    ssind = (79-nlag):(148-nlag);
elseif subsample == 3
    ssind = (149-nlag):(T-nlag);
end;
data_GMM = data_GMM(ssind,:);
Z = Z(ssind,:);

% 1st-stage estimation: Take W = identity 
W = eye(r);
theta_GMM1= fminunc('GMMobjCGG', thetainit, optionsGMM1, data_GMM, Z, W);

% get optimal weighting matrix
W = GMMoptWCGG(theta_GMM1, data_GMM, Z, lambda);

% 2nd-stage estimation:: Take optimal W 
% The second output of fminunc is the minimized objective function.
[theta_GMM2, Q]= fminunc('GMMobjCGG', theta_GMM1, optionsGMM2, data_GMM, Z, W);

% get asymptotic covariance matrix for theta_GMM2
cov_GMM = GMMcovmatCGG(theta_GMM2, data_GMM, Z, W);
%[Q,~,~,~]=GMMobjCGG(theta_GMM2, data_GMM,Z,W);
se_GMM = sqrt(diag(cov_GMM));      % standard error
J = nobs * Q;              % J statistic
prob = 1 - chi2cdf(J, r);  % p-value.

% save results
table_GMM = [theta_GMM2, se_GMM];

%% Step 5: Statistical Inference
% % Hausman test
nparam=size(theta_NLS,1);
COV=([cov_GMM-cov_NLS]\eye(nparam));
H=(theta_NLS-theta_GMM2)'*COV*(theta_NLS-theta_GMM2);
probH = 1 - chi2cdf(H, nparam);  % p-value.


% 95% confidence region wrt. (beta, gamma)
% % choose possible beta's and gamma's according to subsamples
% if subsample == 1
%     betavec = betavec1; gammavec = gammavec1;
% elseif subsample == 2
%     betavec = betavec2; gammavec = gammavec2;
% elseif subsample == 3
%     betavec = betavec3; gammavec = gammavec3;
% end;
%     
% nb = size(betavec,1);    % # of beta's tried
% ng = size(gammavec,1);   % # of gamma's tried
% 
% % compute J statistic for each pair of (beta, gamma)
% % fix rho and pistar at their GMM estimates
% Jmat = zeros(nb,ng);
% for i = 1:nb
%      for j = 1:ng
%           beta = betavec(i);
%           gamma = gammavec(j);
%           theta = [beta; gamma; theta_GMM2(3:4,1)];
%           W = GMMoptWCGG(theta, data_GMM, Z, lambda);
%           Qtemp = GMMobjCGG(theta, data_GMM, Z, W);
%           Jmat(i,j) = nobs * Qtemp;
%      end;    
% end;    
% 
% betamat = kron(betavec, ones(1,ng));     % linescale for beta
% gammamat = kron(gammavec', ones(nb,1));  % linescale for gamma
% 
% figure(1)  % plot 3D figure of all J statistics
% surf(betamat, gammamat, Jmat, 'EdgeColor','none', 'FaceColor','interp', 'FaceLighting','phong');
% xlabel('\beta','FontSize',11); ylabel('\gamma','FontSize',11); 
% zlabel('J stat','FontSize',11); 
% set(gca, 'FontSize', 11);
% 
% figure(2)  % plot contour of all J statistics
% contour(betamat, gammamat, Jmat, 'ShowText', 'on');
% xlabel('\beta','FontSize',11); ylabel('\gamma','FontSize',11); 

