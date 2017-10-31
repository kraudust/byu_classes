function [ output_args ] = ekf_slam(mu_tmin1, sigma_tmin1, u_tmin1, z_t, sigma_r, sigma_phi, Ts, alpha)
    %Table 10.1 in Probabilistic Robotics
    th = mu_tmin1(3);
    vt = u_tmin1(1);
    omegat = u_tmin1(2);
    N = length(mu_tmin1) - 3;
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
    sigma_bart = Gt*sigma_tmin1*Gt.' + Fx.'*Rt*Fx;
    Qt = [sigma_r^2     0;...
        0               sigma_phi^2];
    
    for i = 1:size(z_t,1)/2
        %if the landmark was never seen before
        if 
            
        end
    end





end

