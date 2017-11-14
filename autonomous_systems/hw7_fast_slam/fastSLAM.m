function Y_t = fastSLAM(z_t, ct, u_tm1, Y_tm1)
    global landmarks_seen sigma_r sigma_phi alpha Ts
    M = length(Y_tm1); %number of particles
    Qt = [sigma_r, 0; 0, sigma_phi];
    p0 = 1/M; % initially all particles have equal weights
    for k = 1:M
        xk_tm1 =  Y_tm1.x(k,:); % particle k's state [x;y;theta] at previous time step
        mu_tm1 = Y_tm1.mu; % estimate of all landmarks locations
        sigma_tm1 = Y_tm1.sigma; % estimate of all landmarks covariances
        xk_t = velocity_motion_model(u_tm1, xk_tm1, alpha, Ts); %propegate particle k forward
        
        j = ct;
        r_tj = z_t(j,1);
        phi_tj = z_t(j,2);
        %if a landmark hasn't been seen before
        if landmarks_seen(j) == 0
            % particle k's estimate of landmark j's location (initialize)
            mu_kj_t = xk_t(1:2)' + [r_tj*cos(wrapToPi(phi_tj + xk_t(3))), r_tj*sin(wrapToPi(phi_tj + xk_t(3)))];
            % calculate x and y distance from landmark estimate to particle estimate
            delta = [mu_kj_t(1) - xk_t(1); mu_kj_t(2) - xk_t(2)];
            q = delta'*delta;
            % calculate jacobian (got it from line 16 in table 10.1)
            H = (1/q) * [sqrt(q) * delta(1),    sqrt(q) * delta(2);...
                        -delta(2),              delta(1)];
            sigma_kj_t = H\Qt/(H');
            wk = p0;
            landmarks_seen(j) = 1; %mark this landmark as seen
        else
            % calculate x and y distance from landmark extimate to particle estimate
            delta = [mu_tm1(j, 1) - xk_t(1); mu_tm1(j,2) - xk_t(2)];
            q = delta'*delta;
            % calculate jacobian (got it from line 16 in table 10.1)
            H = (1/q) * [sqrt(q) * delta(1),    sqrt(q) * delta(2);...
                        -delta(2),              delta(1)];
            Q = H*sigma_tm1
                    
        end
    end
end

