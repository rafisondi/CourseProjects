% Kalman Filter design for the Balancing Cube.
%
% - Check stability and observability.
% - Implement time-varying Kalman Filter.
%   Implement steady-state Kalman Filter.
% - Consider different values for Q (process noise).
%
% Course: Recursive Estimation, Spring 2013
% Problem Set: Kalman Filtering
%
% --
% ETH Zurich
% Institute for Dynamic Systems and Control
% S. Trimpe
% strimpe@ethz.ch
% 2013
%
% --
% Revision history
% [14.05.13, ST]    First version
% [20.05.13, PR]    Minor edits

close all;

%% Configuration
% Simulation time:
T = 100;    % []=samples (i.e. simulation time T/100 seconds)

% Process noise variance: user input
disp('Choose process noise variance:');
disp(' (a) Q = 1e-6 I');
disp(' (b) Q = 1e-3 I');
disp(' (c) Q = 1e-9 I');
userInput = input('Enter a, b, or c:','s');

switch userInput
    case 'a'
        Q = 1e-6*eye(16);
    case 'b' 
        Q = 1e-3*eye(16);
    case 'c'
        Q = 1e-9*eye(16);
    otherwise
        error('Wrong user input.');
end;


%% Load Balancing Cube model.
% Contained in sys_d.
% Discrete-time model with sampling rate of 100 Hz.
%
% States:
%   x_1: angle arm 1
%   x_2: angular velocity arm 1
%   x_3: angle arm 2
%   ...
%   x_11: angle arm 6
%   x_12: angular velocity arm 6
%   x_13: cube roll angle
%   x_14: cube roll angular velocity
%   x_15: cube pitch angle
%   x_16: cube pitch angular velocity
%
% Measurements:
%   z_1: angle arm 1
%   z_2: cube roll angular velocity
%   z_3: cube pitch angular velocity
%   z_4: angle arm 2
%   ...
%   z_16: angle arm 6
%   z_17: cube roll angular velocity
%   z_18: cube pitch angular velocity
%
load CubeModel;

A = sys_d.a;
H = sys_d.c;

n = length(A);      % Number of states
n_meas = size(H,1); % Number of measurements
n_arms = 6;         % Number of arms (control agents)


%% Noise and initial state statistics
R = 1e-6*diag(repmat([1 0 0],1,n_arms)) ...  % absolute encoder noise (small number)
    + 2e-5*diag(repmat([0 1 1],1,n_arms));

Q_sq = sqrt(Q);
R_sq = sqrt(R);

x0 = zeros(n,1);
P0 = eye(n)*3e-4;   % standard deviation: ~1 deg (or deg/s)


%% Check system properties
% Stability
disp('# Check stability:');
ev = eig(A);
disp('Absolute value of poles:');
disp(abs(ev)');
disp('The system is unstable (there are poles with magnitude greater 1).');
disp(' ');

% Observability
% We compute the singular values of the observability matrix to check for
% observability.  While we could simply check the rank of the observability
% matrix, looking at the singular values allows us to compare the
% observability of the different modes.  Loosely speaking, a large singular
% value indicates a mode that is "easy" to observe, a small singular values
% indicates a model that is "hard" to observe, and a singular value
% identical to 0 indicates an unobservable mode.
disp('# Check observability:');
sv = svd(obsv(A,H));
disp('Singular values of observability matrix:');
disp(sv');
disp('The system is observable (all singular values greater than 0).');
disp(' ');


%% Design a steady-state Kalman Filter.
% Compute the steady-state Kalman Filter gain from the solution of the
% Discrete Algebraic Riccati Equations.

P = dare(A',H',Q,R);
K_inf = P*H'*inv(H*P*H'+R);

% Steady-state KF A-matrix
A_sskf = (eye(n)-K_inf*H)*A;


%% Simulation and Kalman Filter Implementation
% Variable to store state and state estimate
x = zeros(n,T+1);
x(:,1) = sqrt(P0)*randn(n,1); 

xHat1 = zeros(n,T+1);   % Time-varying KF
xHat1(:,1) = x0;

xHat2 = zeros(n,T+1);   % Steady-state KF
xHat2(:,1) = x0;

% KF variables
Pm = P0;

for k=2:(T+1)
    % Simulate system (uncontrolled)
    x(:,k) = A*x(:,k-1) + Q_sq*randn(16,1);
    z = H*x(:,k) + R_sq*randn(n_meas,1);
    
    % Time-varying Kalman Filter:
    xp = A*xHat1(:,k-1);
    Pp = A*Pm*A' + Q;
    
    K_tv = Pp*H'*inv(H*Pp*H'+R);    % time-varying gain
    xHat1(:,k) = xp + K_tv*(z - H*xp);
    aux = (eye(n)-K_tv*H);
    Pm =  aux*Pp*aux' + K_tv*R*K_tv';
    
    % Steady-state Kalman Filter:
    xHat2(:,k) = A_sskf*xHat2(:,k-1) + K_inf*z;
end;

% Estimation error
err1 = x-xHat1;
err2 = x-xHat2;

%% Plots

% Select what states to plot (select 4).
sel = [1 2 13 14];

% Figure 1: states and state estimates
figure;
for i=1:4
    subplot(4,1,i);
    plot(0:T,x(sel(i),:)/pi*180,0:T,xHat1(sel(i),:)/pi*180,0:T,xHat2(sel(i),:)/pi*180);
    ylabel(['x(',int2str(sel(i)-1),')'])
    grid;
    if i==1
        title('States and state estimates (in deg or deg/s)');
    end;
end;
xlabel('Discrete-time step k');
legend('true state','TVKF estimate','SSKF estimate');


% Figure 2: estimation errors
figure;
for i=1:4
    subplot(4,1,i);
    plot(0:T,err1(sel(i),:)/pi*180,0:T,err2(sel(i),:)/pi*180);
        ylabel(['x(',int2str(sel(i)-1),')'])
    grid;
    if i==1
        title('Estimation error (in deg or deg/s)');
    end;
end;
xlabel('Discrete-time step k');
legend('TVKF estimate','SSKF estimate');


%% Analysis
% Compute the poles of the error dynamics for the steady-state KF.
poles = sort(abs(eig(A_sskf)));
disp(['Magnitude of error dynamic eigenvalues: ']);
disp(poles');

% Compute squared estimation error.
disp(['Squared estimation error.']);
disp(['Time-varying KF: ',num2str(trace(err1*err1'))]);
disp(['Steady-state KF: ',num2str(trace(err2*err2'))]);

