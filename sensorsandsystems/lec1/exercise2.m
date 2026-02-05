%% Exercise 2: Discretization with Forward Euler and Tustin
clear; clc; close all;

s = tf('s');
Ts = 0.5;               % sample time [s]
t_end = 20;
t_c = 0:0.01:t_end;     % continuous-time grid
t_d = 0:Ts:t_end;       % discrete-time grid

%% Continuous-time models
G1 = (0.5*s + 1) / (s^3 - 3*s^2 + 0.4*s + 2);

G2 = [ (0.2*s^2 + 2*s + 1)/(s^2 + 4*s + 1);
       (0.2*s^2 + 1)/(s^2 + 4*s + 1) ];

A3 = [-4 -1; 1 0];
B3 = [2; 0];
C3 = [ 0.6  0.4;
      -0.4  0.4];
D3 = [0; 0];
G3 = ss(A3,B3,C3,D3);

% Convert all to minimal state-space (easier for uniform Euler method)
G1c = minreal(ss(G1));
G2c = minreal(ss(G2));
G3c = minreal(ss(G3));

%% Discretize
% Forward Euler: x[k+1] = (I + Ts*A)x[k] + Ts*B u[k]
G1d_FE = forwardEulerSS(G1c, Ts);
G2d_FE = forwardEulerSS(G2c, Ts);
G3d_FE = forwardEulerSS(G3c, Ts);

% Tustin (bilinear)
G1d_TU = c2d(G1c, Ts, 'tustin');
G2d_TU = c2d(G2c, Ts, 'tustin');
G3d_TU = c2d(G3c, Ts, 'tustin');

%% Stability check
fprintf('--- Stability (continuous / FE / Tustin) ---\n');
fprintf('G1: %d / %d / %d\n', isstable(G1c), isstable(G1d_FE), isstable(G1d_TU));
fprintf('G2: %d / %d / %d\n', isstable(G2c), isstable(G2d_FE), isstable(G2d_TU));
fprintf('G3: %d / %d / %d\n', isstable(G3c), isstable(G3d_FE), isstable(G3d_TU));

%% Pole mapping insight (optional printout)
printPoleMap('G1', G1c, Ts);
printPoleMap('G2', G2c, Ts);
printPoleMap('G3', G3c, Ts);

%% STEP comparison
compareStep('G1 step: continuous vs FE vs Tustin', G1c, G1d_FE, G1d_TU, t_c, t_d);
compareStep('G2 step: continuous vs FE vs Tustin', G2c, G2d_FE, G2d_TU, t_c, t_d);
compareStep('G3 step: continuous vs FE vs Tustin', G3c, G3d_FE, G3d_TU, t_c, t_d);

%% LSIM comparison with same input as before
u_c = sin(0.8*t_c) + 0.5*(t_c >= 5);
u_d = sin(0.8*t_d) + 0.5*(t_d >= 5);

compareLsim('G1 lsim: continuous vs FE vs Tustin', G1c, G1d_FE, G1d_TU, u_c, t_c, u_d, t_d);
compareLsim('G2 lsim: continuous vs FE vs Tustin', G2c, G2d_FE, G2d_TU, u_c, t_c, u_d, t_d);
compareLsim('G3 lsim: continuous vs FE vs Tustin', G3c, G3d_FE, G3d_TU, u_c, t_c, u_d, t_d);

%% ---------- Local functions ----------
function sysd = forwardEulerSS(sysc, Ts)
    [A,B,C,D] = ssdata(sysc);
    n = size(A,1);
    Ad = eye(n) + Ts*A;
    Bd = Ts*B;
    Cd = C;
    Dd = D;
    sysd = ss(Ad,Bd,Cd,Dd,Ts);
end

function compareStep(figTitle, sysc, sysFE, sysTU, tc, td)
    [yc, tc] = step(sysc, tc);
    [yfe, tfe] = step(sysFE, td);
    [ytu, ttu] = step(sysTU, td);

    [ny,~,~] = size(yc); %#ok<ASGLU>
    ny = size(yc,2); % outputs

    figure('Name',figTitle);
    for i = 1:ny
        subplot(ny,1,i);
        yci  = squeeze(yc(:,i,1));
        yfei = squeeze(yfe(:,i,1));
        ytui = squeeze(ytu(:,i,1));

        plot(tc, yci, 'k', 'LineWidth', 1.4); hold on;
        stairs(tfe, yfei, 'b', 'LineWidth', 1.1);
        stairs(ttu, ytui, 'r', 'LineWidth', 1.1);
        grid on;
        ylabel(sprintf('Out(%d)', i));
        if i==1, title(figTitle); end
        if i==ny, xlabel('Time (s)'); end
        legend('Continuous','Forward Euler','Tustin','Location','best');
    end
end

function compareLsim(figTitle, sysc, sysFE, sysTU, uc, tc, ud, td)
    [yc, tc] = lsim(sysc, uc, tc);
    [yfe, tfe] = lsim(sysFE, ud, td);
    [ytu, ttu] = lsim(sysTU, ud, td);

    ny = size(yc,2);
    if ny == 1
        yc = yc(:); yfe = yfe(:); ytu = ytu(:);
    end

    figure('Name',figTitle);
    for i = 1:ny
        subplot(ny,1,i);
        if ny == 1
            yci = yc; yfei = yfe; ytui = ytu;
        else
            yci = yc(:,i); yfei = yfe(:,i); ytui = ytu(:,i);
        end
        plot(tc, yci, 'k', 'LineWidth', 1.4); hold on;
        stairs(tfe, yfei, 'b', 'LineWidth', 1.1);
        stairs(ttu, ytui, 'r', 'LineWidth', 1.1);
        grid on;
        ylabel(sprintf('Out(%d)', i));
        if i==1, title(figTitle); end
        if i==ny, xlabel('Time (s)'); end
        legend('Continuous','Forward Euler','Tustin','Location','best');
    end
end

function printPoleMap(name, sysc, Ts)
    pc = pole(sysc);
    z_exact = exp(pc*Ts);
    z_fe    = 1 + pc*Ts;
    z_tu    = (1 + pc*Ts/2) ./ (1 - pc*Ts/2);

    fprintf('\n%s pole mapping (p -> z)\n', name);
    disp(table(pc, z_exact, z_fe, z_tu, ...
        'VariableNames', {'p_continuous','z_exact_exp','z_forward_euler','z_tustin'}));
end
