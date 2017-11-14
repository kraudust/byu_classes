function Y_t = fastSLAM(z_t, ct, u_tm1, Y_tm1)
    global landmarks_seen sigma_r sigma_phi alpha Ts
    M = length(Y_tm1); %number of particles
    for k = 1:M
        xk_tm1 =  Y_tm1.x(k,:); % particle k's state [x;y;theta] at previous time step
        mu_tm1 = Y_tm1.mu; % estimate of all landmarks locations
        sigma_tm1 = Y_tm1.sigma; % estimate of all landmarks covariances
        xk_t = velocity_motion_model(u_tm1, xk_tm1, alpha, Ts); %propegate particle k forward
        
        j = ct;
        if landmarks_seen(j) == 0
            
        end
    end
end

