clc;
clear all;
data = load('CubeModel.mat');
A = data.sys_d.A;
B = data.sys_d.B;
H = data.sys_d.C;
D = data.sys_d.D;
Obsv = obsv(A,H);
EV = eig(A);
disp("System is unstable, due to Eignvalues being outside of Unit circle");
rank(Obsv);
disp("System is observable, due to full rank");
n_arms = 6;
R = 1e-6*diag(repmat([1 0 0],1,n_arms)) ...  % absolute encoder noise (small number)
    + 2e-5*diag(repmat([0 1 1],1,n_arms));
Q = eye(16) * 1e-6;
P0 = eye(16) * 3e-4 ;
u = zeros(6,1);

%% Generate Dataset / Simulation:
n = 100;
v = mvnrnd(zeros(16,1),Q , n);
w = mvnrnd(zeros(18,1),R , n);
z = zeros(18,n);
x0 = sqrt(P0)*randn(16,1);
x = x0;

x_list = [x0 , zeros(16,n)];
x_m_list = [x0 , zeros(16,n)];
x_ss_list = [x0 , zeros(16,n)];

                             %Assume exact init conditions                           
for  k = (1:n)
        x = A*x + B*u + v(k,:)';
        z(:,k) = H*x + w(k,:)';
        x_list(:,k) = x;
end


%% Kalman Filter


%Steady state Kalman Filter 
P = dare(A',H',Q,R);
K_inf = P*H'*inv(H*P*H'+R);

% Steady-state KF A-matrix
A_sskf = (eye(16)-K_inf*H)*A;

%Init
x_m = x0;
P_m = P0;


for k = (1:n)  
    % Kalman Prior
    x_p = A*x + B*u;
    P_p = A*P_m*A' + Q;
    % Kalman Posterior
    K = P_p*H'*inv(H*P_p*H'+R);
    x_m = x_p + K* (z(:,k) - H*x_p);
    P_m = (eye(16) - K*H) * P_p;
    x_ss = A_sskf*x_p + K_inf*z(:,k);

    x_m_list(:,k) = x_m;
    x_ss_list(:,k) = x_ss;
end

tiledlayout(2,1)
t = (0:n);

% Top plot
ax1 = nexttile;
plot(ax1,t, abs(x_m_list - x_list))
title(ax1,'Top Plot')
ylabel(ax1,'x1')

% Bottom plot
ax2 = nexttile;
plot(ax2,t,abs(x_ss_list - x_list))
title(ax2,'Bottom Plot')
ylabel(ax2,'x2')

