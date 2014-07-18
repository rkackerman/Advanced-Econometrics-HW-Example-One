% function [SSR, dSSR, hess, cov] = NLSobj(theta, data_NLS)
% PURPOSE: compute (the average of) the sum of squared residuals, 
%          its first derivatives wrt. theta, hessian, and the asymptotic
%          covariance matrix 
%--------------------------------------------------------------------
% USAGE: [SSR, dSSR, hess, cov] = NLSobj(theta, data_NLS)
% where: theta     = parameters (nparam x 1)
%        data_NLS  = data matrix (T x nvar)
%        nparam    = # of parameters
%        nvar      = # of variables
%--------------------------------------------------------------------
% RETURNS: SSR   = sum of squared residuals / T (scalar)
%          dSSR  = gradient (nparam x 1)
%          hess  = Hessian (nparam x nparam)
%          cov   = asymptotic covariance matrix for theta_hat (nparam x nparam)
%--------------------------------------------------------------------
% Reference: R. Clarida, J. Gali, & M. Gertler (2000).
%            Monetary Policy Rules and Macroeconomic Stability: Evidence
%            and Some Theory. Quarterly Journal of Economics, 115, 147-180.
% ---------------------------------------------------------------------
% Written by Anonymous
% Updated by Robert Ackerman, UNC Chapel Hill.
% Oct. 15, 2013.
% --------------------------------------------------------------------
   
% parameters
beta = theta(1);
gamma = theta(2);
rho = theta(3);
pistar = theta(4);

nobs = size(data_NLS,1);
nparam = size(theta,1);

% get each series
FF = data_NLS(:,1);
pi1 = data_NLS(:,2);
X1 = data_NLS(:,3);
FF_lag = data_NLS(:,4);

% get epsilon
alpha_ = mean(FF, 1) - beta * pistar;
y = alpha_ + beta * pi1 + gamma * X1;
eps = FF - (1 - rho) * y - rho * FF_lag;

% get individual moment vector and average moment vector
g = eps;
g_bar = mean(g, 1)';
%W = diag(eps'*eps);
% NLS objective function
Q = (1/nobs)*(g'*g);

% derivative of y wrt theta
dy = zeros(nobs, nparam);
dy(:,1) = -pistar + pi1;
dy(:,2) = X1;
dy(:,4) = -beta;

% derivative of epsilon wrt theta
deps = -(1-rho) * dy;
deps(:,3) = y - FF_lag;

% derivative of g wrt theta
dg = deps;
dg_bar = mean(dg, 1);

% derivative of Q wrt theta
dQ = (2/nobs) * (dg' * g);
SSR=Q;
dSSR=dQ;

hess = zeros (nparam, nparam, nobs);
hess(1,3,:) = -pistar + pi1;
hess(1,4,:) = 1-rho;
hess(2,3,:) = X1;
hess(3,1,:) = -pistar + pi1;
hess(3,2,:) = X1;
hess(3,4,:) = -beta;
hess(4,1,:) = 1-rho;
hess(4,3,:) = -beta;

%Get Omegahat and compute cov
Omegahat=var(eps);
cov = ((deps'*deps)\eye(4))*deps'*Omegahat*deps*((deps'*deps)\eye(4));   
