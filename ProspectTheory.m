%% Prospect Theory (Loss Aversion)
%
% Codes solves an infinite horizon consumption-savings problem with stochastic income that displays a prospect theory utility function.
%
% Two main things are required to implement Prospect Theory in VFI Toolkit:
%  First, the lag of consumption is an (endogenous) state variable.
%  Second, is that the prospect theory can then be dealt with in the return function.
%
% These codes just solve the household problem (implement the partial eqm)
%
% This version has endogenous labour (albeit currently set to trivial one grid point, n_l=1, so it is actually exogenous, but can easily be increased to n_l=2 or more).
%
%% Before the code, let's look at the problem to be solved.
%
% Prospect Theory, also called Loss Aversion, is the idea that people dislike losses more than they like gains.
% So the utility of getting \$100 is less than the utility of losing \$100. This requires a reference point relative 
% to which outcomes are judged to be a gain or loss. In this implementation we follow a literature that sets 
% 'last period consumption' as the reference point for gains/losses of this periods consumption outcome.
% [See Köbberling & Wakker (2004) - An index of loss aversion. 
% Or especially page 25 of Santoro, Petrella, Pfajfar & Gaffeo (2014) - Loss aversion and the asymmetric transmission of monetary policy.]
% 
% Let $u(c)=\frac{c^{1-sigma}}{1-sigma}$ be a standard CES period-utility function. [CES is constant elasticity of substitution]
% Where $c$ is this period consumption. 
% Let $clag$ be last period consumption.
% We define $Delta=c-clag$. So when $Delta>0$ there is a gain, and when $Delta<0$ there is a loss.
% 
% The prospect theory period-utility function is given by
%    $U(c)=\theta u(c)+ (1-\theta) v(u(c)-u(clag))$
% where $u(c)$ is a standard period-utility function (in our case the CES period-utility fn)
% and $v(Delta)$ implements the loss aversion taking different forms depending on the sign of Delta, specifically
%    $v(Delta)=(1-e^{-\mu*Delta})/\mu$ for $Delta>=0$
%    $v(Delta)=-\lambda*(1-e^{(\upsilon/\lambda)*Delta)}/\upsilon$ for $Delta<0$
% 
% The prospect-theory parameters play the following roles. $\theta$ is the (inverse of the) importance of the loss aversion, relative 
% to the standard utility. $\mu$ controls how quickly the sensitivity of gains decreases at the margin with larger gains (similarly for 
% $\upsilon$ and losses). $\lambda$ indexes the degree of loss aversion, effectively determining the importance of losses relative to gains.
%
% The full utility function of the agent/household is the infinite-horizon present discounted expected utility, using the prospect-theory period-utility function U(c).
%
% In principle, the below also allows for endogenous labour (if you set n_l>1) in which case a $-\chi \frac{l^{1+\eta}}{1+\eta}$ term is added to
% the period-utility function. $l$ is the labour supply, and $\eta$ is the Frisch elasticity of labour supply. $\chi$ determines the disutility of
% working relative to the utility of consumption.

addpath(genpath('./MatlabToolkits/'))

%% Set the grid sizes
n_l=1; % Endogenous labour
n_assets=251; % Asset holdings
n_clag=101; % Lag of consumption (which determines the reference level for prospect theory)
n_z=5; % Exogenous shock, efficiency labour units

%% Set the parameters

% Preferences
Params.beta=0.96; % Discount factor

% Propect theory parameters
Params.theta=0.5;
Params.mu=1;
Params.upsilon=1;
Params.lambda=2.25;
Params.sigma=1.5;
% I use the following notation
% (1-theta)*u(c_t)+theta*v(u(c_t)-u(c_t-1))
% where v(Delta)=(1-e^(-mu*Delta))/mu for Delta>=0
%       v(Delta)=-lambda*(1-e^((upsilon/lambda)*Delta))/upsilon for Delta<0
% I set
% u(c)=(c^(1-sigma))/(1-sigma)

% Labor/leisure preference parameters
Params.chi=1;
Params.eta=0.25;

% Prices
Params.r=0.04; % Interest rate (this interest rate should be interpreted as net of depreciation)
Params.w=1; % Wage rate

% Exogenous labor efficiency units process
Params.rho_z=0.9;
Params.sigma_epsilonz=0.04;
% Tauchen method hyperparameter
Params.q=2;

%% Set the grids
l_grid=linspace(0,1,n_l)'; % Note that when n_l=1 this just gives l_grid=1
assets_grid=25*(linspace(0,1,n_assets).^3)';
clag_grid=linspace(0,4,n_clag)';
Params.cgridspacing=clag_grid(2)-clag_grid(1); % The spacing of the clag_grid has to be the same between any two consecutive points (is assumed by code in the return fn)

[z_grid, pi_z]=TauchenMethod(0,Params.sigma_epsilonz,Params.rho_z,n_z,Params.q); %[states, transmatrix]=TauchenMethod_Param(mew,sigmasq,rho,znum,q,Parallel,Verbose), transmatix is (z,zprime)
z_grid=exp(z_grid);
% Make it so that E[z]=1
[Expectation_z,~,~,~]=MarkovChainMoments(z_grid,pi_z);
z_grid=z_grid./Expectation_z;
[Expectation_z,~,~,~]=MarkovChainMoments(z_grid,pi_z);

%% Put into VFI toolkit notation
n_d=n_l;
n_a=[n_assets,n_clag];
% n_z=n_z;
d_grid=l_grid;
a_grid=[assets_grid; clag_grid];
% z_grid=z_grid

%% Set the discount factor and return function
DiscountFactorParamNames={'beta'};

ReturnFn=@(l_val,assetsprime_val,clagprime_val,assets_val,clag_val,z_val,r,w,sigma,theta,mu,lambda,upsilon,chi,eta,cgridspacing) ProspectTheory_ReturnFn(l_val,assetsprime_val,clagprime_val,assets_val,clag_val,z_val,r,w,sigma,theta,mu,lambda,upsilon,chi,eta,cgridspacing);
ReturnFnParams={'r','w','sigma','theta','mu','lambda','upsilon','chi','eta','cgridspacing'}; %It is important that these are in same order as they appear in 'ProspectTheory_ReturnFn'

%% Solve
%Do the value function iteration. Returns both the value function itself, and the optimal policy function.
tic;
vfoptions.lowmemory=1 % 1 means loop over z (instead of parallelizing), this reduces memory usage but increases runtime.
[V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParams,vfoptions);
vftime=toc;

%%
simoptions.parallel=3; % Sparse matrices on cpu
StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions);

%% Report some output
fprintf('Run time for value function iteration: %8.2f seconds \n', vftime)

%% Graph of the value function (for the median grid point of clag; note that this may not be hugely relevant choice for clag)
figure(1)
surf(assets_grid*ones(1,n_z), ones(n_assets,1)*z_grid', reshape(V(:,floor(n_clag/2),:),[n_assets,n_z]))
title('Value Function')
saveas(gcf,'./SavedOutput/Graphs/Fig_ValueFn.png')

%% Graph of the policy function (for the median grid point of clag; note that this may not be hugely relevant choice for clag)
figure(2)
plot(l_grid(permute(Policy(1,:,floor(n_clag/2),:),[2,4,1,3])))
title('Leisure choice')
saveas(gcf,'./SavedOutput/Graphs/Fig_Choice_Leisure.png')

figure(3)
plot(a_grid(permute(Policy(2,:,floor(n_clag/2),:),[2,4,1,3])))
title('Asset choice')
saveas(gcf,'./SavedOutput/Graphs/Fig_Choice_Asset.png')

figure(4)
plot(clag_grid(shiftdim(Policy(3,:,:),1)))
title('(Next period) Consumption (lag) choice')
saveas(gcf,'./SavedOutput/Graphs/Fig_Choice_Consumption.png')


%% How does the prospect theory compare to the same model but with just a CES utility function.

% We need to drop clag (or more accurately, it is much easier and the codes will be much faster)
n_a2=n_assets;
a2_grid=assets_grid;
% Change the return function to CES utility fn
ReturnFn2=@(l_val,assetsprime_val,assets_val,z_val,r,w,sigma,chi,eta) CES_ReturnFn(l_val,assetsprime_val,assets_val,z_val,r,w,sigma,chi,eta);
ReturnFnParams2={'r','w','sigma','chi','eta'}; %It is important that these are in same order as they appear in 'ProspectTheory_ReturnFn'

tic;
vfoptions.lowmemory=1;
[V_CES,Policy_CES]=ValueFnIter_Case1(n_d,n_a2,n_z,d_grid,a2_grid,z_grid, pi_z, ReturnFn2, Params, DiscountFactorParamNames, ReturnFnParams2,vfoptions);
vftime=toc;

simoptions.parallel=3; % Sparse matrices on cpu
StationaryDist_CES=StationaryDist_Case1(Policy_CES,n_d,n_a2,n_z,pi_z, simoptions);

% Comparing the actual numbers of the value function is meaningless (as utility is only defined up to a linear transformation; e.g., adding a constant 
% to the utility function changes the numbers but in no way changes the preferences/decisions)

% So let's compare the asset and leisure choices.
figure(5)
plot(l_grid(permute(Policy(1,:,floor(n_clag/2),:),[2,4,1,3])))
hold on
plot(l_grid(shiftdim(Policy_CES(1,:,:),1)))
hold off
% Following line assumes that n_z=5. If you change n_z then it will error.
legend('Prospect, z_c=1','Prospect, z_c=2','Prospect, z_c=3','Prospect, z_c=4','Prospect, z_c=5','CES, z_c=1','CES, z_c=2','CES, z_c=3','CES, z_c=4','CES, z_c=5')
title('Leisure choice')
saveas(gcf,'./SavedOutput/Graphs/Fig_Compare_Leisure.png')


figure(6)
plot(a_grid(permute(Policy(2,:,floor(n_clag/2),:),[2,4,1,3])))
hold on
plot(a_grid(shiftdim(Policy_CES(2,:,:),1)))
hold off
% Following line assumes that n_z=5. If you change n_z then it will error.
legend('Prospect, z_c=1','Prospect, z_c=2','Prospect, z_c=3','Prospect, z_c=4','Prospect, z_c=5','CES, z_c=1','CES, z_c=2','CES, z_c=3','CES, z_c=4','CES, z_c=5')
title('Asset choice')
saveas(gcf,'./SavedOutput/Graphs/Fig_Compare_AssetChoice.png')

FnsToEvaluateParamNames(1).Names={'r','w'};
FnsToEvaluateFn_c = @(l_val,assetsprime_val,assets_val,z_val,r,w) (1+r)*assets_val+w*l_val*z_val-assetsprime_val; % Consumption
FnsToEvaluate={FnsToEvaluateFn_c};

ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_Case1(StationaryDist_CES, Policy_CES, FnsToEvaluate, Params, FnsToEvaluateParamNames, n_d, n_a2, n_z, d_grid, a2_grid, z_grid);

% To be able to compare the consumption choices we need to first calculate consumption for the CES.
figure(7)
plot(clag_grid(shiftdim(Policy(3,:,:),1))) % Note that ploting the choice of next period consumption lag is effectively just ploting current consumption
hold on
plot(shiftdim(ValuesOnGrid(1,:,:),1))
hold off
legend('Prospect, z_c=1','Prospect, z_c=2','Prospect, z_c=3','Prospect, z_c=4','Prospect, z_c=5','CES, z_c=1','CES, z_c=2','CES, z_c=3','CES, z_c=4','CES, z_c=5')
title('Consumption choice')
saveas(gcf,'./SavedOutput/Graphs/Fig_Compare_Consumption.png')


%% We can also compare the stationary distribution of agents for prospect theory vs CES. The following lines do that.

figure(8)
plot(cumsum(sum(sum(StationaryDist,3),2)))
hold on
plot(cumsum(sum(StationaryDist_CES,2)))
hold off
legend('Prospect','CES')
title('CDF over Assets')
saveas(gcf,'./SavedOutput/Graphs/Fig_AssetCDF.png')

% Note that this graph is not the Lorenz curve but is not far from it

%% I thought it would be interesting to look at the C/Y and K/Y ratios and how they differ with prospect theory.

% For Prospect Theory
FnsToEvaluateParamNames(1).Names={};
FnsToEvaluateFn_c = @(l_val,assetsprime_val,clagprime_val,assets_val,clag_val,z_val) clagprime_val; % Consumption
FnsToEvaluateParamNames(2).Names={'r','w'};
FnsToEvaluateFn_income = @(l_val,assetsprime_val,clagprime_val,assets_val,clag_val,z_val,r,w) r*assets_val+w*l_val*z_val; % Income
FnsToEvaluateParamNames(3).Names={};
FnsToEvaluateFn_assets = @(l_val,assetsprime_val,clagprime_val,assets_val,clag_val,z_val) assets_val; % Assets
FnsToEvaluate={FnsToEvaluateFn_c,FnsToEvaluateFn_income,FnsToEvaluateFn_assets};

AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate,Params, FnsToEvaluateParamNames,n_d, n_a, n_z, d_grid, a_grid,z_grid);

% For CES
FnsToEvaluateParamNames(1).Names={'r','w'};
FnsToEvaluateFn_c = @(l_val,assetsprime_val,assets_val,z_val,r,w) (1+r)*assets_val+w*l_val*z_val-assetsprime_val; % Consumption
FnsToEvaluateParamNames(2).Names={'r','w'};
FnsToEvaluateFn_income = @(l_val,assetsprime_val,assets_val,z_val,r,w) r*assets_val+w*l_val*z_val; % Income
FnsToEvaluateParamNames(3).Names={};
FnsToEvaluateFn_assets = @(l_val,assetsprime_val,assets_val,z_val) assets_val; % Assets
FnsToEvaluate={FnsToEvaluateFn_c,FnsToEvaluateFn_income,FnsToEvaluateFn_assets};

AggVars_CES=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_CES, Policy_CES, FnsToEvaluate,Params, FnsToEvaluateParamNames,n_d, n_a2, n_z, d_grid, a2_grid,z_grid);

fprintf('The C/Y ratio is %8.3f for Prospect Theory and %8.3f for CES \n',AggVars(1)/AggVars(2),AggVars_CES(1)/AggVars_CES(2))
fprintf('The K/Y ratio is %8.3f for Prospect Theory and %8.3f for CES \n',AggVars(3)/AggVars(2),AggVars_CES(3)/AggVars_CES(2))

%%
save ./SavedOutput/ProspectTheory.mat V V_CES Policy Policy_CES StationaryDist StationaryDist_CES AggVars AggVars_CES ValuesOnGrid

% load ./SavedOutput/ProspectTheory.mat V V_CES Policy Policy_CES StationaryDist StationaryDist_CES AggVars AggVars_CES ValuesOnGrid



