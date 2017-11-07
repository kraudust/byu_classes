function [mu_t, sigma_t] = ekf_slam(mu_tmin1, sigma_tmin1, u_tmin1, z_t, lm)
    global landmarks_seen sigma_r sigma_phi alpha Ts
    %Table 10.1 in Probabilistic Robotics
    th = mu_tmin1(3);
    vt = u_tmin1(1);
    omegat = u_tmin1(2);
    N = (length(mu_tmin1) - 3)/2; %number of landmarks
    Fx = [eye(3), zeros(3,2*N)];
    mu_tbar = mu_tmin1 + Fx.'* ...
        [-(vt/omegat)*sin(th) + (vt/omegat)*sin(th + omegat*Ts);...
            (vt/omegat)*cos(th) - (vt/omegat)*cos(th + omegat*Ts);...
            omegat*Ts];
    Gt = eye(3+2*N) + Fx.'*...
        [...
            0   0   -(vt/omegat)*cos(th) + (vt/omegat)*cos(th + omegat*Ts);...
            0   0   -(vt/omegat)*sin(th) + (vt/omegat)*sin(th + omegat*Ts);...
            0   0                           0]* Fx;
    Mt = [alpha(1)*vt^2 + alpha(2)*omegat^2     0;...
            0                                   alpha(3)*vt^2 + alpha(4)*omegat^2];
    Vt = [...
        (-sin(th) + sin(th + omegat*Ts))/omegat     vt*(sin(th) - sin(th + omegat*Ts))/(omegat^2) + vt*cos(th + omegat*Ts)*Ts/omegat;...
        (cos(th) - cos(th + omegat*Ts))/omegat      -vt*(cos(th) - cos(th + omegat*Ts))/(omegat^2) + vt*sin(th + omegat*Ts)*Ts/omegat;...
        0                                           Ts];
    Rt = Vt*Mt*Vt.';
    sigma_tbar = Gt*sigma_tmin1*Gt.' + Fx.'*Rt*Fx;
    Qt = [sigma_r^2     0;...
        0               sigma_phi^2];
    
    for i = 1:N
        z_ti = [z_t(i);z_t(i+N)]; %z_t is [R1;R2;R3;phi1;phi2;phi3]
        r_ti = z_ti(1);
        phi_ti = z_ti(2);
        %figure out if landmark is in the field of view
        if abs(wrapToPi(phi_ti)) < pi/4
                %j = i;
            %if the landmark was never seen before, initialize it
            if landmarks_seen(i) == 0
                mu_tbar(4 + 2*(i-1)) = mu_tbar(1) + r_ti*cos(phi_ti + mu_tbar(3));
                mu_tbar(5 + 2*(i-1)) = mu_tbar(2) + r_ti*sin(phi_ti + mu_tbar(3));
                landmarks_seen(i) = 1;
            end
            deltax = mu_tbar(4 + 2*(i-1)) - mu_tbar(1);
            deltay = mu_tbar(5 + 2*(i-1)) - mu_tbar(2);
            delta = [deltax; deltay];
            q = delta.'*delta;
            z_thati = [...
                sqrt(q);...
                atan2(deltay, deltax) - mu_tbar(3)];
            F_xj = [...
                eye(3),     zeros(3,2*i - 2), zeros(3,2), zeros(3,2*N-2*i);...
                zeros(2,3), zeros(2,2*i - 2), eye(2), zeros(2,2*N-2*i)];
            H_ti = (1/q)*...
                [-sqrt(q)*deltax, -sqrt(q)*deltay, 0, sqrt(q)*deltax, sqrt(q)*deltay;...
                deltay, -deltax, -q, -deltay, deltax]*...
                F_xj;
            K_ti = sigma_tbar*H_ti.'/(H_ti*sigma_tbar*H_ti.' + Qt);
            mu_tbar = mu_tbar + K_ti*wrapToPi(z_ti - z_thati);
            sigma_tbar = (eye(3 + 2*N) - K_ti*H_ti)*sigma_tbar;
        end
    end
    mu_t = mu_tbar;
    sigma_t = sigma_tbar;
end

