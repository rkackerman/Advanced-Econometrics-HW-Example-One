
% Robert Ackerman
% Homework 2
% Problem 2
% October 15, 2013

% L. P. Hansen and K. J. Singleton (1982). Generalized Instrumental
% Variables Estimation of Nonlinear Rational Expectations Models,
% Econometrica, 50, 1269-1286.

% See also: GMMobj.m, GMMcovmat.m, GMMoptW.m, NWHAC.m.
%%
clc;
clear;
load 'HW2Q2b.txt';
%Set range of sample 1=Feb1959 239=Dec1978 651=Apr2013
first = 1;
last = 239;
%set instrument lags
nlag = 1;
%Use NWHAC.m to get Newy West 1987 HAC estimator
lambda = 'NW';

% grid for alpha beta first stage est (note: picking range based on what we
% think is the likely alpha and beta)
avec1 = (-1:0.01:0)';
bvec1 = (0.9:0.01:1.1)';

% grid for alpha beta second stage est (note: increment is smaller here
avec2 = (-1:0.001:-0.5)';
bvec2 = (0.9:0.001:1.1)';
%% PT2 Grid search
X = HW2Q2b(first:last, :);   % get x1 and x2
T = size(X,1);                  % sample size
na1 = size(avec1,1);            % # of alpha's tried in 1st stage
nb1 = size(bvec1,1);            % # of beta's tried in 1st stage
na2 = size(avec2,1);            % # of alpha's tried in 2nd stage
nb2 = size(bvec2,1);            % # of beta's tried in 2nd stage

% 1st stage
Q = Inf;                        % start with infinite obj. fnc. and update
for i = 1:na1      % loop wrt. alpha, CRRA parameter
     alpha = avec1(i);
     for j = 1:nb1  % loop wrt. beta, discount factor
          beta = bvec1(j);
          theta = [alpha; beta];  
          Q_temp = GMMobj(X, theta, nlag); % 1st stage with identity weighting matrix
         
          if Q_temp < Q      % if new theta improves Q, then save it.
                             % Otherwise keep the previous Q and theta.
              Q = Q_temp;
              GMME = [alpha; beta];
          end;    
     end;
end;

W = GMMOptW(X, GMME, nlag, lambda);
%%
% Second stage using the optimal weighting matrix
Q2 = zeros(na2, nb2);  % save all candidate values to draw a 3D figure

% 2nd stage
Q = Inf;                        % start with infinite obj. fnc. and update
for i = 1:na2      % loop wrt. alpha, CRRA parameter
     alpha = avec2(i);
     for j = 1:nb2  % loop wrt. beta, discount factor
          beta = bvec2(j);
          theta = [alpha; beta];  
          Q_temp = GMMobj(X, theta, nlag, W); % 2nd stage with optimal weighting matrix
          Q2(i,j)=Q_temp;
          if Q_temp < Q      % if new theta improves Q, then save it.
                             % Otherwise keep the previous Q and theta.
              Q = Q_temp;
              GMME = [alpha; beta];
          end;    
     end;
end;
          

%% Step 3: Statistical Inference
cov = GMMcovmat(X, GMME, nlag, W);
se = sqrt(diag(cov));   % standard error
J = T * Q;              % J statistic
dof = 2 * nlag - 1;     % degrees of freedom 
                        % # of moment restrictions = 2*nlag + 1
                        % # of parameters = 2
prob = chi2cdf(J, dof);  % p-value

% make Table 1 in p.1282
table = [nlag; GMME(1); se(1); GMME(2); se(2); J; dof; prob];

% plot objective function for each pair of (alpha, beta)
amat = kron(avec2, ones(1,nb2));   % linescale for alpha
bmat = kron(bvec2', ones(na2,1));  % linescale for beta
figure(1)  % plot 3D figure
surf(amat, bmat, Q2, 'EdgeColor','none', 'FaceColor','interp', 'FaceLighting','phong');
xlabel('\alpha','FontSize',11); ylabel('\beta','FontSize',11); 
zlabel('Q','FontSize',11); 
set(gca, 'FontSize', 11);

% clear original data
 clear HW2Q2b
