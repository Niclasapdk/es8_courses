function res = HoKalmanId(hk, param)
    p = param.p;   % number of outputs
    r = param.r;   % number of inputs
    u = param.u;   % number of block rows (past)
    n = param.n;   % desired system order

    % dummy variables for system matrices
    Aest = [];
    Cest = [];
    Best = [];

    %% Build the Block Hankel Matrix H(1) and H(2)
    % hk is assumed to be a 3D array: hk(:,:,k) = Markov parameter H_k
    % size: [p x r x numSamples]

    numCols = u;   % number of block columns
    numRows = u;   % number of block rows (can differ, but square is common)

    % H(1): starts at Markov parameter index 1
    H1 = zeros(p * numRows, r * numCols);
    for i = 1:numRows
        for j = 1:numCols
            H1((i-1)*p+1 : i*p, (j-1)*r+1 : j*r) = hk(:,:, i+j-1);
        end
    end

    % H(2): starts at Markov parameter index 2 (shift by 1)
    H2 = zeros(p * numRows, r * numCols);
    for i = 1:numRows
        for j = 1:numCols
            H2((i-1)*p+1 : i*p, (j-1)*r+1 : j*r) = hk(:,:, i+j);
        end
    end

    %% SVD of H1
    [U, S, V] = svd(H1);

    %% Truncate to desired order n (plot singular values to choose n)
    Un = U(:, 1:n);
    Sn = S(1:n, 1:n);
    Vn = V(:, 1:n);

    %% Extract system matrices
    % Observability matrix  : O = Un * Sn^(1/2)
    % Controllability matrix: C = Sn^(1/2) * Vn'

    Sn_sqrt     = diag(sqrt(diag(Sn)));
    Sn_sqrt_inv = diag(1 ./ sqrt(diag(Sn)));

    O = Un * Sn_sqrt;   % [p*u x n]  observability matrix
    Ct = Sn_sqrt * Vn'; % [n x r*u]  controllability matrix

    %% Recover C, B, A
    Cest = O(1:p, :);                    % first p rows of O
    Best = Ct(:, 1:r);                   % first r cols of controllability matrix

    % A from the shifted Hankel: A = Sn^(-1/2) * Un' * H2 * Vn * Sn^(-1/2)
    Aest = Sn_sqrt_inv * Un' * H2 * Vn * Sn_sqrt_inv;

    %% Output
    res.A = Aest;
    res.B = Best;
    res.C = Cest;
    res.S = diag(Sn);   % singular values (useful for order selection)
end