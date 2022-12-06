clc;
clear all;

%% Simulate system:
mu = 3.9;
alpha = 0.07;
a = 0.6;
b = 0.15;
N = 200;
x_list = zeros(2,N+1);
z_list = zeros(2,N);

%Init
x = rand;
y = rand;
x_list(:,1) = [x y]';
for k = 2:N+1

    %Model
    l_x = x*(1-x)*mu;
    l_y = y*(1-y)*mu;
    v_x = rand * a;
    v_y = rand * a;
    x = (1-alpha) * l_x +(1-v_y) * alpha * l_y;
    y = (1-alpha) * l_y +(1-v_x) * alpha * l_x;

    %Measurement
    w_x = rand * b;
    w_y = rand * b;
    z1 = x * (1-w_x);
    z2 = y * (1-w_y);
    
    x_list(:,k) = [x y]';
    z_list(:,k-1) = [z1 z2]'; 
end

%% Extended Kalman Filter implementation:
%Init 
x_m = [ 0.5 0.5]';
x_p = zeros(2,1);
P_m = eye(2)*1/12;
Q = eye(2)*a^2/12; % process noise variance, identical for v1 and v2
R = eye(2)*b^2/12; % measurement noise variance, identical for w1, w2

x_m_list = zeros(2,N+1);
x_m_list(:,1) = x_m;

for k = 1:N
%Recalculate system matrices


 A = [(1-alpha)*mu*(1- 2*x_m(1))         ,       (1-a/2)*alpha*mu*(1-2*x_m(2)); 
      (1-a/2)*alpha * mu*(1 - 2* x_m(1)) ,       (1-alpha)*mu* (1- 2*x_m(2)) ] ; 
% 
 L = [0 , -alpha * mu * x_m(2) * (1 -x_m(2));
      -alpha * mu * x_m(1) * (1 -x_m(1)) , 0];

q_x = (1-alpha)* (x_m(1)*(1-x_m(1))*mu) + (1-a/2)*alpha*(x_m(2)*(1-x_m(2))*mu );
q_y = (1-alpha)* (x_m(2)*(1-x_m(2))*mu) + (1-a/2)*alpha*(x_m(1)*(1-x_m(1))*mu );


%Kalman Filter
x_p = [q_x q_y]';
P_p = A*P_m*A' + L*Q*L';

H = eye(2)*(1-b/2);
M =-diag(x_p);

h_x = x_p(1)*(1- b/2);
h_y = x_p(2)*(1- b/2);


K = P_p*H'*inv(H*P_p*H' + M*R*M');
x_m =  x_p + K * (z_list(:,k) - (1- b/2)*x_p);
%P_m = (eye(2) - K * H)*P_p*(eye(2) - K * H)' + K *M* R *M'* K';
P_m = (eye(2) - K * H)*P_p;

x_m_list(:,k+1) = x_m;
end

tiledlayout(2,1)
t = (0:N);
% Top plot
ax1 = nexttile;
plot(ax1,t,x_list(1,:), t,x_m_list(1,:))
title(ax1,'Top Plot')
ylabel(ax1,'x')

% Bottom plot
ax2 = nexttile;
plot(ax2, t,x_list(2,:),t,x_m_list(2,:))
title(ax2,'Bottom Plot')
ylabel(ax2,'y')
