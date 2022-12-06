clear all
clc

%% Exercise 1 b) 

u_vtg = 0.5;
u_egr = 0.5;
[sysC, linearizationPt] = getLinearModel(u_vtg, u_egr);

%Discretize system:
Ts = 0.05;
sysD = c2d(sysC,Ts, 'zoh');

%Use MPC Controller:
N = 50; %Horizon
[Gamma,Psi,Upsilon,Theta] = setupPredictionMatrices(sysD.A,sysD.B,sysD.C,N);

%% Exercise 2a) Optimal control law K_mpc
% Create Weighting matrixes Q and R 
    Q_i = diag([10^(-10) 10]);
    R_i = diag([10 10]);

Q = kron(eye(N), Q_i);
R = kron(eye(N), R_i);

%Calculate K_mpc
B = sysD.B;
l = size(B,2);
    M = zeros(l, l*N);
    M(1:l, 1:l) = eye(l);
K_mpc = M*((Theta'*Gamma'*Q*Gamma*Theta+R)\(Theta'*Gamma'*Q)); % including only first delta_u


% reference signals
  reference.time = (0:20:100)';
  reference.p_im_ref = [1.05 1.05 1.05 1.15 1.05 1.05]'*1e5;
  reference.x_bg_ref = [0.1 0.16 0.1 0.1 0.1 0.1]';

% Simulink reference format: structure    
  referenceSimulink.time               = reference.time;
  referenceSimulink.signals.values     = [reference.p_im_ref, reference.x_bg_ref];
  referenceSimulink.signals.dimensions = size(referenceSimulink.signals.values,2);
  
% linear model    
  model = 'ps04_ex3_LinearModel_t';
  sim(model, reference.time([1 end]));

  disp("All done")