%**************************************************************************
%   Problem Set 04, Exercise 3                                            *
%   Model Predictive Engine Control                                       *
%   Spring 2022, IDSC                                                     *
%**************************************************************************

    
%% Linear Unconstrained MPC Controller: Instructions
% 
%  Make sure, your linear model "sysC" from Exercise 1, which was 
%  linearized at the operation point "linearizationPt" exists in the 
%  workspace. Also, all the matrices derived in Exercise 1 and 2 have to be
%  available in the workspace.
%
%  Open the models "ps04_ex3_LinearModel.slx" and "ps04_ex3_ROM.slx" and
%  implement your MPC controller.
%
%  Run this file to check your controller and plot the results.
%
%% Linear Unconstrained MPC Controller Evaluation with Linear Model & ROM
% 1) linear model
% 2) ROM - nonlinear

% reference signals
  reference.time = (0:20:100)';
  reference.p_im_ref = [1.05 1.05 1.05 1.15 1.05 1.05]'*1e5;
  reference.x_bg_ref = [0.1 0.16 0.1 0.1 0.1 0.1]';

% Simulink reference format: structure    
  referenceSimulink.time               = reference.time;
  referenceSimulink.signals.values     = [reference.p_im_ref, reference.x_bg_ref];
  referenceSimulink.signals.dimensions = size(referenceSimulink.signals.values,2);
    
% linear model    
  model = 'ps04_ex3_LinearModel';
  sim(model, reference.time([1 end]));

% nonlinear ROM    
  model = 'ps04_ex3_ROM';
  sim(model, reference.time([1 end]));    
    
%% plot simulation results and measurements

mfig = 2;
nfig = 2;
cfig = 1;
figure
 ax(cfig) = subplot(mfig,nfig,cfig); cfig = cfig +1;
    stairs(reference.time,reference.p_im_ref,'--k'); hold on; grid on;
    plot(simOutput_LinMod.p_im.Time,simOutput_LinMod.p_im.Data, 'LineWidth', 2)
    plot(simOutput_ROM.p_im.time,simOutput_ROM.p_im.Data, 'LineWidth', 2)
    xlabel('time [s]'); ylabel('p\_im [Pa]')
    legend('reference', 'LinMod', 'ROM')
  ax(cfig) = subplot(mfig,nfig,cfig); cfig = cfig +1;
  stairs(reference.time,reference.x_bg_ref,'--k'); hold on; grid on;
    plot(simOutput_LinMod.x_bg.Time,simOutput_LinMod.x_bg.Data, 'LineWidth', 2)   
    plot(simOutput_ROM.x_bg.Time,simOutput_ROM.x_bg.Data, 'LineWidth', 2)
    xlabel('time [s]'); ylabel('x\_bg [-]')
    legend('reference', 'LinMod', 'ROM')
  ax(cfig) = subplot(mfig,nfig,cfig); cfig = cfig +1;
    plot(simOutput_LinMod.u_vtg.Time,simOutput_LinMod.u_vtg.Data, 'LineWidth', 2); hold on; grid on;
    plot(simOutput_ROM.u_vtg.Time,simOutput_ROM.u_vtg.Data, 'LineWidth', 2)
    xlabel('time [s]'); ylabel('u\_vtg [-]')
    legend('LinMod', 'ROM')
  ax(cfig) = subplot(mfig,nfig,cfig); cfig = cfig +1;
    plot(simOutput_LinMod.u_egr.Time,simOutput_LinMod.u_egr.Data, 'LineWidth', 2); hold on; grid on;
    plot(simOutput_ROM.u_egr.Time,simOutput_ROM.u_egr.Data, 'LineWidth', 2)
    xlabel('time [s]'); ylabel('u\_egr [-]')
    legend('LinMod', 'ROM')      
    
    