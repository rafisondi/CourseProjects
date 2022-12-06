% Time-Varying Kalman Filter design for the Water Tank System
%
% Course: Recursive Estimation, Spring 2013
% Problem Set: Kalman Filtering
%
% --
% ETH Zurich
% Institute for Dynamic Systems and Control
% Markus Hehn
% hehnm@ethz.ch
% 2013
%
% --
% Revision history
% [13.05.13, MH]    First version
% [20.05.13, PR]    Revision
%
clc; 
clear all;
% Simulation settings:
% Number of steps.
N = 1000;

% Choose part of the problem: C through G.
problemPart = 'G';

%% Simulate the real system
% Distribution parameters
x0 = 5*ones(3,1);
P0 = eye(3);
Q = diag([1/40, 1/10, 1/5]);
R = 0.5*eye(2);

% Draw samples of the process noise for all time steps:
v = (randn(N,3)*chol(Q))';
   
% Vector for full state history:
x = zeros(3,N+1);
% Draw random initial condition:
x(:,1) = x0 + chol(P0)*randn(3,1);

% Control input:
if(problemPart == 'D')
    u = 5*abs(sin(1:N));
else
    u = 5*ones(1,N);
end

% State dynamics:
% Parameters:
alpha1 = 0.1;
alpha2 = 0.5;
alpha3 = 0.2;
    
% B matrix is constant in all problem parts:
B = [0.5;0.5;0];

% A matrix is potentially time-varying. Ak contains a full history of A
% for all cases:
if(problemPart == 'C' || problemPart == 'D' || problemPart == 'E')
    A = [1-alpha1, 0, 0; 0, 1, 0; alpha1, 0, 1-alpha3];
    Ak = repmat(A, [1 1 N]);
elseif(problemPart == 'F')
    A = [1-alpha1, 0, 0; 0, 1-0.5, 0; alpha1, 0.5, 1-alpha3];
    Ak = repmat(A, [1 1 N]);
elseif(problemPart == 'G')
    A = [1-alpha1, 0, 0; 0, 1, 0; alpha1, 0, 1-alpha3];
    Ak = repmat(A, [1 1 N]);
    Ak(2,2,1:3:N) = 1-0.5;
    Ak(3,2,1:3:N) = 0.5;
end

% Measurements:
% Vector for full measurement history:
z = zeros(2,N+1);

% H matrix depends on problem part:
if(problemPart == 'C' || problemPart == 'D')
    H = [0 1 0; 0 0 1];
else
    H = [1 0 0; 0 0 1];
end

% Draw samples of the measurement noise for all time steps:
w = (randn(N+1,2)*chol(R))';

% Compute state and measurement history from Ak, B, u, v, H, w:
for k = 2:(N+1)
    x(:,k) = Ak(:,:,k-1)*x(:,k-1)+B*u(k-1)+v(:,k-1);
    z(:,k) = H*x(:,k)+w(:,k);
end

%% Run the Kalman Filter.

% Initial mean and variance:
x0_hat = x0;
P0_hat = P0;

% Activate for part c3: random initial PSD, diagonal variance:
drawRandomVar = 0;
if(drawRandomVar)
    P0_hat = diag(5*rand(3,1).*diag(P0));
end

% Mean and variance history vectors:
x_hat = zeros(3,N+1);
P_hat = zeros(3,3,N+1);
P_p = zeros(3,3,N+1);

% Set initial mean and variance.
% problem part A.
x_hat(:,1) = x0_hat;
P_hat(:,:,1) = P0_hat;

% Pre-compute the inverse of R because we need it all the time.
R_inv = inv(R);

% Execute the Kalman Filter equations for each time step:
% Matlab index starts at 1, which corresponds to k = 0, i.e. P_p(:,:,2) is P_p(1)
for i = 2:(N+1)
    x_p = Ak(:,:,i-1)*x_hat(:,i-1)+B*u(i-1);
    P_p(:,:,i) = Ak(:,:,i-1)*P_hat(:,:,i-1)*Ak(:,:,i-1)' + Q;
    
    P_hat(:,:,i) = inv(inv(P_p(:,:,i)) + H'*R_inv*H);
    x_hat(:,i) = x_p + P_hat(:,:,i)*H'*R_inv*(z(:,i)-H*x_p);
end

%% Plot results
%Mean
figure(1)
for i = 1:3
    subplot(3,1,i)
    plot(0:N, x(i,:), 'k-', 0:N, x_hat(i,:), 'b--', ...
        0:N, x_hat(i,:)'+sqrt(squeeze(P_hat(i,i,:))), 'r--',...
        0:N, x_hat(i,:)'-sqrt(squeeze(P_hat(i,i,:))), 'r--')
    legend('True state', 'Estimated state', '+/- 1 standard deviation');
    ylabel(['Tank ' num2str(i) ' Level, x(',int2str(i),')'])
    xlabel('Time step k')
end
% Variances
figure(2)
plot(1:N, squeeze(P_p(1:3,1,2:end)), 1:N, squeeze(P_p(2:3,2,2:end)), ...
    1:N, squeeze(P_p(3,3,2:end)));
xlabel('Time step k')
ylabel('Covariance matrix entry value')
legend('P_{p11}', 'P_{p12}','P_{p13}','P_{p22}','P_{p23}','P_{p33}')