function [K,R] = gl2kr(A,G,C,L0)
    [P,flag] = solvric(A,G,C,L0);
    if (flag == 1)
        disp('Warning: Non positive real covariance model => K = R = []');
        K = [];
        R = [];
    else
        % Make output (Page 63 for instance)
        R = L0 - C*P*C';
        K = (G - A*P*C')/(R);
    end
    %
    function [P,flag] = solvric(A,G,C,L0)
        if isempty(L0) || isempty(G)
            P = [];
            flag = 0;
        else
            [~,n_n]=size(A); 			% Dimensions
            L0i=inv(L0); 				% Compute the inverse once
            % Set up the matrices for the eigenvalue decomposition
            AA = [A'-C'*L0i*G' zeros(n_n,n_n);-G*L0i*G' eye(n_n)];
            BB = [eye(n_n) -C'*L0i*C;zeros(n_n,n_n) A-G*L0i*C];
            % Compute the eigenvalue decomposition
            [v,d] = eig(AA,BB);
            ew = diag(d);
            % If there's an eigenvalue on the unit circle => no solution
            flag = 0;
            if all(abs(abs(ew)-ones(2*n_n,1)) > 1e-9) < 1;flag = 1;end
            % Sort the eigenvalues
            [~,II]=sort(abs(ew));
            % Compute P
            P=real(v(n_n+1:2*n_n,II(1:n_n))/v(1:n_n,II(1:n_n)));
        end
    end
end
