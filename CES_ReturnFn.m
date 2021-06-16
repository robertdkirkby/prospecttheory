function F=CES_ReturnFn(l_val,assetsprime_val,assets_val,z_val,r,w,sigma,chi,eta)

F=-Inf;

c=(1+r)*assets_val+w*l_val*z_val-assetsprime_val;

% I set
% u(c)=(c^(1-sigma))/(1-sigma)

if c>0
    F_c=(c^(1-sigma))/(1-sigma);

    F_l=-chi*(l_val^(1+eta))/(1+eta); % Disutility of leisure
    
    F=F_c+F_l;
end

end