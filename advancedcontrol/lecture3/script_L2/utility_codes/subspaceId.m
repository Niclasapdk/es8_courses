function res = subspaceId(hk,param)

    p = param.p;
    r = param.r;
    m = param.m;
    n = param.n;
    mthd = param.mthd; 

    % dummy variables for system matrices
    Aest = [];
    Cest = [];
    Xest = [];

    %% general for all algorithms

    %% start coding : svd of the block Hankel matrix
    
    [U,S,V] = svd(hk,'econ');
    S = diag(S);    
    
    %% stop coding 

    % once the singluar vectors are obtained, one can estimate system
    % matrices

    %% start coding : estimating A,C

        Us = U(:,1:n);
        Ss = S(1:n);
        Vs = V(:,1:n);
    
        Ob = Us*diag(sqrt(Ss));
    
        Ob_up  =  Ob(1:p*r,:);
        Ob_dn = Ob(r+1:(p+1)*r,:);
    
        Aest = pinv(Ob_up)*Ob_dn;
        Cest  =  Ob(1:r,:);

    %% stop coding 
        
    %% computing UPC states 
    %% start coding 

    switch mthd

        case 'UPC'
        
            hkm = param.hkm; 

            
    
    end


    %%
    
    res.Aest = Aest;
    res.Cest = Cest;
    res.Xest = Xest;
    




end
