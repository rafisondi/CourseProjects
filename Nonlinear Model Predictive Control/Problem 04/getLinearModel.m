function [linearModel, linearizationPt] = getLinearModel(u_vtg_val, u_egr_val)

%   GETLINEARMODEL returns linearized Reduced Order Model linearModel and 
%   the corresponding linearization point linearizationPt
%   The nonlinear Reduced Order Model is linearized using the symbolic 
%   toolbox of Matlab.

% input variables
    syms u_vtg u_egr 
% state variables
    syms n_tc p_im p_em F_im F_em 

% xdot = f(x,u)
    [xdot, y] = ROModel([n_tc; p_im; p_em; F_im; F_em], [u_vtg; u_egr]);   
% symbolic state space matrices
    symbROM.A = jacobian([xdot(1), xdot(2), xdot(3), xdot(4), xdot(5)],[n_tc, p_im, p_em, F_im, F_em]);
    symbROM.B = jacobian([xdot(1), xdot(2), xdot(3), xdot(4), xdot(5)],[u_vtg, u_egr]);
    symbROM.C = jacobian([y(1), y(2)],[n_tc, p_im, p_em, F_im, F_em]);
    symbROM.D = jacobian([y(1), y(2)],[u_vtg, u_egr]);

% so far everything was symbolic, evaluate now with numeric values
% define input variables    
    u_vtg = u_vtg_val;
    u_egr = u_egr_val;
% get values of states at linearization point
    u = [u_vtg; u_egr];
    ic = [8e3; 1.21e5; 1.50e5; 0.23; 0.16];
    options = optimset('Display','off');
    ss_vals = fsolve(@(x) ROModel(x,u), ic', options);    
    n_tc = ss_vals(1);
    p_im = ss_vals(2);
    p_em = ss_vals(3);
    F_im = ss_vals(4);
    F_em = ss_vals(5);
% state space description    
    A = eval(subs(symbROM.A)); 
    B = eval(subs(symbROM.B)); 
    C = eval(subs(symbROM.C)); 
    D = eval(subs(symbROM.D));

% create output structs
% linearModel
    linearModel = ss(A,B,C,D);
    linearModel.InputName = {'u_vtg','u_egr'};
    linearModel.StateName = {'n_tc','p_im','p_em','F_im','F_em'};
    linearModel.OutputName = {'p_im','x_bg'};
% linearizationPt
    linearizationPt.uOP = [u_vtg_val; u_egr_val];
    linearizationPt.xOP = ss_vals';
    [~, linearizationPt.yOP] = ROModel(linearizationPt.xOP, linearizationPt.uOP);
    
end
