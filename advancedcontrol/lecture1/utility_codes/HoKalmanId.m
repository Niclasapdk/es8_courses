function res = HoKalmanId(hk, param)
    p = param.p;   % number of block rows
    r = param.r;   % number of outputs
    u = param.u;   % number of inputs
    n = param.n;   % desired system order
    
    % hk is already the block Hankel matrix from SSImat
    % Expected dimensions: [r*(p+1) x numCols]
    
    [rows, cols] = size(hk);
    
    % Verify dimensions - should be r*(p+1) for this formulation
    expected_rows = r * (p + 1);
    if rows ~= expected_rows
        warning('Row dimension: expected %d (r*(p+1)), got %d. Adjusting p accordingly.', expected_rows, rows);
        p = floor(rows / r) - 1;
        param.p = p;
    end
    
    %% SVD of the Hankel matrix
    [U, S, V] = svd(hk, 'econ');
    S = diag(S);  % Extract singular values as a vector
    
    %% Truncate to desired order n
    if n > length(S)
        warning('Requested order n=%d is larger than available rank %d. Using n=%d instead.', n, length(S), length(S));
        n = length(S);
    end
    
    Us = U(:, 1:n);
    Ss = S(1:n);
    Vs = V(:, 1:n);
    
    %% Build observability matrix
    Ob = Us * diag(sqrt(Ss));
    
    %% Extract shifted observability matrices
    Ob_up = Ob(1:p*r, :);                % Upper part: rows 1 to p*r
    Ob_dn = Ob(r+1:(p+1)*r, :);          % Lower part: rows r+1 to (p+1)*r (shifted)
    
    %% Estimate system matrices
    Aest = pinv(Ob_up) * Ob_dn;          % State matrix using pseudoinverse
    Cest = Ob(1:r, :);                   % Output matrix: first r rows
    
    %% Estimate B matrix using controllability
    % Controllability matrix from SVD
    Reac = diag(sqrt(Ss)) * Vs';         % Reachability matrix
    Best = Reac(:, 1:u);                 % First u columns
    
    %% Compute condition number of eigenvalues
    eigenvalues = eig(Aest);
    
    % Condition number can be computed in different ways:
    % 1. Ratio of largest to smallest magnitude
    eigen_magnitudes = abs(eigenvalues);
    if min(eigen_magnitudes) > 1e-10  % Avoid division by near-zero
        cond_number_magnitude = max(eigen_magnitudes) / min(eigen_magnitudes);
    else
        cond_number_magnitude = Inf;
    end
    
    % 2. Condition number of eigenvector matrix (more robust measure)
    [V_eig, D_eig] = eig(Aest);
    cond_number_eigenvectors = cond(V_eig);
    
    %% Output
    res.A = Aest;
    res.B = Best;
    res.C = Cest;
    res.Aest = Aest;              % For compatibility
    res.Best = Best;
    res.Cest = Cest;
    res.S = Ss;                   % Singular values
    res.eigenvalues = eigenvalues;
    res.cond_magnitude = cond_number_magnitude;
    res.cond_eigenvectors = cond_number_eigenvectors;
    
    % Display results
    fprintf('\n=== System Identification Results ===\n');
    fprintf('System order n = %d\n', n);
    fprintf('Number of singular values retained: %d\n', length(Ss));
    fprintf('\nSingular values (top %d):\n', min(10, length(Ss)));
    disp(Ss(1:min(10, length(Ss)))');
    
    fprintf('\nEigenvalues of A:\n');
    disp(eigenvalues);
    
    fprintf('\nCondition numbers:\n');
    fprintf('  Magnitude ratio (max/min): %.4e\n', cond_number_magnitude);
    fprintf('  Eigenvector matrix cond():  %.4e\n', cond_number_eigenvectors);
    
    % Stability check
    max_eigen_mag = max(abs(eigenvalues));
    if max_eigen_mag < 1
        fprintf('\nSystem is STABLE (max |eigenvalue| = %.4f < 1)\n', max_eigen_mag);
    else
        fprintf('\nSystem is UNSTABLE (max |eigenvalue| = %.4f >= 1)\n', max_eigen_mag);
    end
    fprintf('=====================================\n\n');
end