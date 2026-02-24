function [hk,param]=SSImat(inMat,param)
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

    hkm = [];
    
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

        case 'OOcov'
            pq = p + q;

            nj=(p+1)*r + q*r;    
            RQ = zeros(N-pq, nj);  
            for i=0:q-1 
                RQ(:,i*r+(1:r)) = inMat(q-1-i+(1:N-pq),:);	%Y-
            end        
            for i=0:p
                RQ(:,q*r+i*r+(1:r)) = inMat(i+q+(1:N-pq),:);	%Y+
            end

            hk = RQ(:,(q*r+1):nj)'*RQ(:,1:q*r) / (N-pq); % Y+Y-^T

        case 'UPC'
            pq = p + q;

            nj=(p+1)*r + q*r;    
            RQ = zeros(N-pq, nj);  
            for i=0:q-1 
                RQ(:,i*r+(1:r)) = inMat(q-1-i+(1:N-pq),:);	%Y-
            end        
            for i=0:p
                RQ(:,q*r+i*r+(1:r)) = inMat(i+q+(1:N-pq),:);	%Y+
            end
            RQ = RQ./(N-pq); % normalizing with the number of samples

            hk = RQ(:,(q*r+1):nj)'*RQ(:,1:q*r)*...
                pinv(RQ(:,1:q*r)'*RQ(:,1:q*r))*RQ(:,1:q*r)'; % Y+Y-^T (Y-Y-^T)^-1 Y- 
            %
            param.hkm = RQ(:,((q+1)*r+1):nj)'*RQ(:,1:(q+1)*r)*...
                pinv(RQ(:,1:(q+1)*r)'*RQ(:,1:(q+1)*r))*RQ(:,1:(q+1)*r)';

    end

end

