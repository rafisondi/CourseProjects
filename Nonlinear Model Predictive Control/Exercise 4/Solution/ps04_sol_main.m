%**************************************************************************
%   Problem Set 04, Exercises 1-3                                          *
%   Model Predictive Engine Control                                       *
%   Spring 2022, IDSC                                                     *
%**************************************************************************

clc
clear
close all
addpath('../../providedCode')

%% 1 b): Derive the continuous-time linear model

  [sysC, linearizationPt] = getLinearModel(0.5, 0.5);

%% 1 c): Derive the discrete-time linear model

  sysD = c2d(sysC, 0.05, 'zoh');

%% 1 d): Test prediction matrix function

  % define MPC horizon
    N = 50;
    
  % get MPC matricies
    [Gamma, Psi, Upsilon, Theta] = ...
        setupPredictionMatrices(sysD.A, sysD.B, sysD.C, N);

  % system dimensions
    n = size(sysD.A,2);        % length of model state vector x
    l = size(sysD.B,2);        % length of model input vector u
    m = size(sysD.C,1);        % length of model output vector y

  % check dimension of Gamma
    fprintf('\nCheck dimenstion of Gamma:\n')
    fprintf('\tExpected dimension: %ix%i\n', [m*N n*N])
    fprintf('\tActual dimension: \t%ix%i\n', size(Gamma))

  % check dimension of Psi
    fprintf('\nCheck dimenstion of Psi:\n')
    fprintf('\tExpected dimension: %ix%i\n', [n*N n])
    fprintf('\tActual dimension: \t%ix%i\n', size(Psi))

  % check dimension of Upsilon
    fprintf('\nCheck dimenstion of Upsilon:\n')
    fprintf('\tExpected dimension: %ix%i\n', [n*N l])
    fprintf('\tActual dimension: \t%ix%i\n', size(Upsilon))
    
  % check dimension of Theta
    fprintf('\nCheck dimenstion of Theta:\n')
    fprintf('\tExpected dimension: %ix%i\n', [n*N l*N])
    fprintf('\tActual dimension: \t%ix%i\n', size(Theta))

%% 2 a): Calculate Q and R from Qi and Ri

  % define weighting matricies Qi and Ri
    Qi = diag([10^(-10) 10]);
    Ri = diag([10 10]);

  % calculate Q and R
    Q = kron(eye(N), Qi);
    R = kron(eye(N), Ri);

%% 2 b): Calculate K_MPC

  % define the matrix M in order to extract the first entries of du( . | k )
    M = zeros(l, l*N);
    M(1:l, 1:l) = eye(l);

  % calculate K_MPC
    K_MPC = M*((Theta'*Gamma'*Q*Gamma*Theta+R)\(Theta'*Gamma'*Q));

%% 3: Run provided matlab script
  
  run ps04_run_ex3
  
    