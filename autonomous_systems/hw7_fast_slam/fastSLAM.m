function [Y_t, w] = fastSLAM(z_t, ct, u_tm1, Y_tm1)
    global landmarks_seen sigma_r sigma_phi alpha Ts
    M = length(Y_tm1.x); %number of particles
    Qt = [sigma_r, 0; 0, sigma_phi];
    p0 = 1/M; % initially all particles have equal weights
    j = ct;
    r_tj = z_t(j,1);
    phi_tj = z_t(j,2);
    fov = pi*2;
    no_part = false;
    while abs(wrapToPi(phi_tj)) > fov/2
        j = j + 1;
        if j > size(landmarks_seen, 2)
            j = 1;
        end
        r_tj = z_t(j,1);
        phi_tj = z_t(j,2);
        if j == ct
            no_part = true; %no particles in field of view
            phi_tj = 0;
        end
    end
    if not(no_part) % if there are particles seen do the full algorithm
        for k = 1:M
            xk_tm1 =  Y_tm1.x(k,:)'; % particle k's state [x;y;theta] at previous time step
            mu_k_tm1 = Y_tm1.mu(:,:,k); % estimate of all landmarks locations from kth particle
            sigma_k_tm1 = Y_tm1.sigma(:,:,k); % estimate of all landmarks covariances from kth particle
            xk_t = velocity_motion_model(u_tm1, xk_tm1, alpha, Ts); %propegate particle k forward
    %         xk_t(3) = wrapToPi(xk_t(3));  
            for j = 1:size(landmarks_seen,2)
                r_tj = z_t(j,1);
                phi_tj = z_t(j,2);
                z_jt = [r_tj; phi_tj];
                if abs(wrapToPi(phi_tj)) < fov/2
                   %if a landmark hasn't been seen before
                    if landmarks_seen(k,j) == 0
                        % particle k's estimate of landmark j's location (initialize)
                        mu_kj_t = xk_t(1:2)' + [r_tj*cos(wrapToPi(phi_tj + xk_t(3))), r_tj*sin(wrapToPi(phi_tj + xk_t(3)))];
                        mu(j,:,k) = mu_kj_t;
                        % calculate x and y distance from landmark estimate to particle estimate
                        delta = [mu_kj_t(1) - xk_t(1); mu_kj_t(2) - xk_t(2)];
                        q = delta'*delta;
                        % calculate jacobian (got it from line 16 in table 10.1)
                        H = (1/q) * [sqrt(q) * delta(1),    sqrt(q) * delta(2);...
                                    -delta(2),              delta(1)];
                        sigma_kj_t = H\Qt/(H');
                        sigma(2*j-1:2*j, :, k) = sigma_kj_t;
                        wj(j) = p0;
                        landmarks_seen(k,j) = 1; %mark this landmark as seen
                    else
                        % calculate x and y distance from landmark estimate to particle estimate
                        delta = [mu_k_tm1(j, 1) - xk_t(1); mu_k_tm1(j,2) - xk_t(2)];
                        q = delta'*delta;

                        % calculate zhat
                        z_hat = [sqrt(q);...
                                wrapToPi(atan2(delta(2), delta(1)) - xk_t(3))];

                        % calculate jacobian (got it from line 16 in table 10.1)
                        H = (1/q) * [sqrt(q) * delta(1),    sqrt(q) * delta(2);...
                                    -delta(2),              delta(1)];
                        Q = H*sigma_k_tm1(2*j - 1: 2*j, :)*H' + Qt; % measurement covariance
                        K = sigma_k_tm1(2*j - 1: 2*j, :) * H'/Q; %calculate kalman gain
                        % mean and covariance of jth landmark from particle k
                        zdiff = z_jt - z_hat;
                        zdiff(2) = wrapToPi(zdiff(2));
                        mu_kj_t = mu_k_tm1(j,:)' + K*zdiff;
                        mu(j,:,k) = mu_kj_t;
                        sigma_kj_t = (eye(2) - K*H)*sigma_k_tm1(2*j - 1: 2*j, :);
                        sigma(2*j-1:2*j, :, k) = sigma_kj_t;
                        wj(j) = ((det(2*pi*Q))^0.5)* exp(-0.5* zdiff' / Q * zdiff); % weight of particle k for landmark j
                    end 
                else
                    mu(j,:,k) = mu_k_tm1(j,:);
                    sigma(2*j-1:2*j, :, k) = sigma_k_tm1(2*j-1:2*j, :);
                end
            end
            
            %calculate particle weight
            %wj = wj./sum(wj);
%             w(k) = dot(wj, wj);
            w(k) = mean(wj);
            
            x(k,:) = xk_t';        
        end
        %Resample particles
        %normalize weights
        w = w./sum(w);
        idx = low_variance_sampler(w);
        Y_t.mu = mu(:,:,idx);
        Y_t.sigma = sigma(:,:,idx);
        Y_t.x = x(idx,:);
    else % if no particles are seen, then just propegate the particles
        for k = 1:M
            xk_tm1 =  Y_tm1.x(k,:)'; % particle k's state [x;y;theta] at previous time step
            xk_t = velocity_motion_model(u_tm1, xk_tm1, alpha, Ts); %propegate particle k forward
            x(k,:) = xk_t'; 
        end
        Y_t.x = x;
        Y_t.mu = Y_tm1.mu;
        Y_t.sigma = Y_tm1.sigma;
        w = p0*ones(1,M);
    end
end

