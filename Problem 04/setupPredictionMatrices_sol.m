function [Gamma,Psi,Upsilon,Theta] = setupPredictionMatrices(A,B,C,N)

%% Get x,u, and y vector sizes

    n = size(A,2);        % length of model state vector x
    l = size(B,2);        % length of model input vector u
    m = size(C,1);        % length of model output vector y

%% Define system response trajectory matrices
  % For a discrete-time system with the state-space matrices A, B, and C, 
  % define the matrices to calculate the system output for a predefined 
  % horizon: y(.|k) = Gamma*(Psi*x(k) + Upsilon*u(k-1) + Theta*du(.|k))
  
% Derive the Gamma matrix
    Gamma = kron(eye(N),C);
    
% Derive the Psi matrix
    Psi = zeros(n*N,n);
    for i = 1:N
        Psi((i-1)*n+1:i*n,:) = A^i;
    end

% Derive the Upsilon matrix
    Upsilon    = zeros(n*N,l);
    UpsilonSum = zeros(size(A));
    for i = 1:N
        UpsilonSum = UpsilonSum + A^(i-1);
        Upsilon((i-1)*n+1:i*n,:) = UpsilonSum*B;
    end

% Derive the Theta matrix
    Theta = zeros(n*N,l*N);
    Theta(1:n,1:l) = Upsilon(1:n,:);
    for i = 2:N
      % Copy last row to new row and shift it
        Theta((i-1)*n+1:i*n,l+1:end) = Theta((i-2)*n+1:(i-1)*n,1:end-l);
      % Set first Line Entry
        Theta((i-1)*n+1:i*n,1:l) = Upsilon((i-1)*n+1:i*n,:);
    end
    
end