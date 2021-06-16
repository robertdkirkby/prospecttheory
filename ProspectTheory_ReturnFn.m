function F=ProspectTheory_ReturnFn(l_val,assetsprime_val,clagprime_val,assets_val,clag_val,z_val,r,w,sigma,theta,mu,lambda,upsilon,chi,eta,cgridspacing)

F=-Inf;

c=(1+r)*assets_val+w*l_val*z_val-assetsprime_val;

% I use the following notation
% (1-theta)*u(c_t)+theta*v(u(c_t)-u(c_t-1))
% where v(Delta)=(1-e^(-mu*Delta))/mu for Delta>=0
%       v(Delta)=-lambda*(1-e^((upsilon/lambda)*Delta))/upsilon
% I set
% u(c)=(c^(1-sigma))/(1-sigma)

if c>0
    Delta=(c^(1-sigma))/(1-sigma)-(clag_val^(1-sigma))/(1-sigma);
    if Delta>=0
        vDelta=(1-exp(-mu*Delta))/mu;
    else % Delta<0
        vDelta=-lambda*(1-exp((upsilon/lambda)*Delta))/upsilon;
    end
    F_c=(1-theta)*(c^(1-sigma))/(1-sigma)+theta*vDelta; % Utility of consumption (including prospect theory/loss aversion)

    F_l=-chi*(l_val^(1+eta))/(1+eta); % Disutility of leisure
    
    F=F_c+F_l;
end

% The other thing that remains to be enforced is that clagprime_val is the same as c (to within grid tolerance)
if abs(c-clagprime_val)>cgridspacing/2
    F=-Inf;
end

end