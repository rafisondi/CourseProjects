clear all;
clc;

% System Matrices;
A = [0.9 , 0 , 0 ;
     0.0 ,0.5,0.0; 
     0.1, 0.5 , 0.8];
A2 = [0.9 , 0 , 0 ;
     0.0 ,1,0.0; 
     0.1, 0 , 0.8];
B = [0.5, 0.5, 0.0]';
H = [1 0 0 ; 
     0 0 1 ];
Q = diag([1/40 , 1/10, 1/5]);            %Process noise
R = diag([0.5 , 0.5]);                   %Measurement noise
%% b) Solve DARE
P_inf = dare(A',H',Q,R);

%% Generate Dataset / Simulation:
n = 2000;
v = mvnrnd([0 0 0]',Q , n);
w = mvnrnd([0 0]',R , n);
z = zeros(2,n);
x0 = [5 5 5]'; 
x = x0;
P0 = eye(3);                             %Assume exact init conditions
u = 5;                                   %Const Input 
x_list = zeros(3,n);
for  k = (1:n)
    if mod(k,2) == 0
        x = A*x + B*u + v(k,:)';
        z(:,k) = H*x + w(k,:)';
    else
        x = A2*x + B*u + v(k,:)';
        z(:,k) = H*x + w(k,:)';
    end
    x_list(:,k) = x;
end

%% Kalman Filter implementation

x_p = 0;
P_p = 0;
K =0;

%Init
x_m = x0;
P_m = 0;
x_m_list = zeros(3,n);

for k = (1:n)  
        % Kalman Prior

        if mod(k,2) == 0
            x_p = A*x + B*u;
            P_p = A*P_m*A' + Q;
        else
            x_p = A2*x + B*u;
            P_p = A2*P_m*A2' + Q;
        end
        
        % Kalman Posterior
        K = P_p*H'*inv(H*P_p*H' +R);
        x_m = x_p + K* (z(:,k) - H*x_p);
        P_m = (eye(3) - K*H) * P_p;
        x_m_list(:,k) = x_m;
end

disp('Pm: -->')
disp(P_m)

disp('Pp: -->')
disp(P_p)

disp('P_infinity: -->')
disp(P_inf)

tiledlayout(3,1)
t = (0:n);
X = [x0 x_m_list];
X_true = [x0 x_list];
% Top plot
ax1 = nexttile;
plot(ax1,t,X(1,:) , t,X_true (1,:))
title(ax1,'Top Plot')
ylabel(ax1,'x1')

% Bottom plot
ax2 = nexttile;
plot(ax2,t,X(2,:),  t,X_true (2,:))
title(ax2,'Bottom Plot')
ylabel(ax2,'x2')

% Bottom plot
ax3 = nexttile;
plot(ax3,t,X(3,:), t,X_true (3,:))
title(ax3,'Bottom Plot')
ylabel(ax3,'x3')

