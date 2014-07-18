
function cov=GMMcovmat(X, GMME, nlag, OptW)
% PURPOSE: compute the asymptotic covariance matrix of two-stage GMM
% estimates
%--------------------------------------------------------------------
% USAGE: cov = GMMcovmat(X, GMME, nlag, OptW)
% where: X   = data matrix (T x 2)
%                 X(:,1) is x1, while X(:,2) is x2.
%       GMME = GMM estimates from second-stage
%     OptW = Optimal weighting matrix
%--------------------------------------------------------------------
% RETURNS: cov   = the asymptotic covariance matrix of two-stage GMM
% estimates
%--------------------------------------------------------------------
% References:  L. P. Hansen and K. J. Singleton (1982). Generalized Instrumental
% Variables Estimation of Nonlinear Rational Expectations Models,
% Econometrica, 50, 1269-1286.
% -------------------------------------------------------------------
% Written by Anonymous
% Updated by Robert Ackerman, UNC Chapel Hill.
% Oct. 14, 2013.
% Also see NWHAC.m, GMMOptW.m and GMMobj.m
%
T = size(X,1);     % sample size   
[Q, f, fderiv] = GMMobj(X, GMME, nlag, OptW, 1);
G=mean(fderiv,3);
cov=1/T*inv(G'*OptW*G);
----------------------------------------------------------------    