function OptW=GMMOptW(X, GMME, nlag, lambda)
% PURPOSE: compute the optimal GMM weighting matrix for second-stage of two-stage GMM
%--------------------------------------------------------------------
% USAGE: OptW = GMMOptW(X, GMME, nlag, lambda)
% where: X    = data matrix (T x 2)
%                 X(:,1) is x1, while X(:,2) is x2.
%        GMME = GMM estimates from first-stage
%      lambda = tuning parameter for Newey and West's (1987) HAC
%                 'NW' for Newey and West's (1994) automatic selection
%--------------------------------------------------------------------
% RETURNS: OptW   = the optimal GMM weighting matrix for second-stage of two-stage GMM
%--------------------------------------------------------------------
% References:  L. P. Hansen and K. J. Singleton (1982). Generalized Instrumental
% Variables Estimation of Nonlinear Rational Expectations Models,
% Econometrica, 50, 1269-1286.
% -------------------------------------------------------------------
% Written by Anonymous
% Updated by Robert Ackerman, UNC Chapel Hill.
% Oct. 14, 2013.
% Also see NWHAC.m and GMMobj.m
% --------------------------------------------------------------------

[Q Z]=GMMobj(X, GMME, nlag);
Sigma=NWHAC(Z,lambda);
OptW=inv(Sigma);
end
