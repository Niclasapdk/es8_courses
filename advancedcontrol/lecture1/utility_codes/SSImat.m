function hk=SSImat(inMat,param)
    %============================================================
    %  Computation of the Hankel matrix with different methods
    %============================================================
    % N : numner of samples
    % p : number of block rows
    % hk : Hankel matrix
    % r : number of sensors
    % inMat : data samples of size(N,r)

    mthd=param.mthd;
    p=param.p;
    q=p+1;

    N=size(inMat,1);
    r=size(inMat,2); 
    
    if r>N
        inMat = inMat.';
        N=size(inMat,1);
        r=size(inMat,2);     
    end

    %******************************************************************
    switch mthd
        case 'impulse response'
            pq = q;

            nj=(p+1)*r;    
            RQ = zeros(N-pq, nj);
            for i=1:q 
                RQ(:,(i-1)*r+1:i*r) = inMat(i+1:N-pq+i,:);	%Y-, shifted by one sample 
            end        

            hk = RQ.';

    end

end

