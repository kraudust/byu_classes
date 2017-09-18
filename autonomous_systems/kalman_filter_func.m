function [mu_t, sigma_t, K_t, sigma_bart] = kalman_filter_func(mu_tm1, sigma_tm1, u_t, z_t, A_t, B_t, C_t, R_t, Q_t)
%KALMAN_FILTER_FUNC Calculates the kalman filter output
%   INPUTS:     mu_tm1-     estimate of states at prior time step
%               sigma_tm1-  estimate of covariance at prior time step
%               u_t-        input
%               z_t-        measurement
%               A_t-        state space A matrix (discrete)
%               B_t-        state space B matrix (discrete)
%               C_t-        state space C matrix (discrete)
%               R_t-        processs covariance
%               Q_t-        measurement covariance
%   OUTPUTS:    mu_t-       new estimate of states
%               sigma_t-    new estimate of covariance

mu_bart = A_t*mu_tm1 + B_t*u_t;
sigma_bart = A_t*sigma_tm1*A_t' + R_t;
K_t = sigma_bart*C_t'*inv(C_t*sigma_bart*C_t' + Q_t);
% Measurement Update
mu_t = mu_bart + K_t*(z_t - C_t*mu_bart);
sigma_t = (eye(length(sigma_tm1)) - K_t*C_t)*sigma_bart;
end

