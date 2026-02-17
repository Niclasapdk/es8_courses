function res = HoKalmanId(hk,param)

    p = param.p;
    r = param.r;
    u = param.u;
    n = param.n;

    
    % dummy variables for system matrices
    Aest = [];
    Cest = [];
    Best = [];

    %% start coding : svd of the block Hankel matrix
    


    
    %% stop coding 

    % once the singluar vectors are obtained, one can estimate system
    % matrices

    %% start coding : estimating A,B,C for the singular vectors 


    %% stop coding 
        



    
    res.Aest = Aest;
    res.Cest = Cest;
    res.Best = Best;
    

end


