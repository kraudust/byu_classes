function [mu_t, sigma_t, Kt] = unscented_kalman_filter(mu_tm1, sigma_tm1, u_tm1, z_t, m, sigma_r, sigma_phi,Ts,alpha_u)
    L = size(mu_tm1,1) + size(u_tm1,1) + size(z_t,1);
    n = L;
    vt = u_tm1(1);
    omegat = u_tm1(2);
    mx = m(1);
    my = m(2);
    
    %Generate augmented mean and covariance
    Mt = [...
        alpha_u(1)*vt^2 + alpha_u(2)*omegat^2,                  0;...
        0,                                                  alpha_u(3)*vt^2 + alpha_u(4)*omegat^2]; %control noise covariance
    
    Qt = [...
        sigma_r^2,              0;...
        0                       sigma_phi^2]; %measurement noise covariance
    
    mu_a_tm1 = [mu_tm1; zeros(size(u_tm1,1),1); zeros(size(z_t,1),1)]; %augmented state
    
    sigma_a_tm1 = [...
        sigma_tm1,                              zeros(size(sigma_tm1,1),size(Mt,2)),    zeros(size(sigma_tm1,1),size(Qt,2));...
        zeros(size(Mt,1),size(sigma_tm1,2)),    Mt,                                     zeros(size(Mt,1),size(Qt,2));...
        zeros(size(Qt,1),size(sigma_tm1,2)),    zeros(size(Qt,1),size(Mt,2)),           Qt]; %augmented covariance
    
    %Generate sigma points (2L + 1)
    R = chol(sigma_a_tm1); % matrix square root
    alpha = 1; %scaling parameter to determine how spread out sigma points are
    kappa = 3; %scaling parameter to determine how spread out sigma points are
    lamda = alpha^2 * (n + kappa) - n;
    gamma = sqrt(n+lamda);
    chi_a_tm1 = [mu_a_tm1, repmat(mu_a_tm1,1,L) + gamma*R, repmat(mu_a_tm1,1,L) - gamma*R];
    chi_x_tm1 = chi_a_tm1(1:3,:);
    chi_u_t = chi_a_tm1(4:5, :);
    chi_z_t = chi_a_tm1(6:7,:);
    
    %Pass sigma points through motion model and compute gaussian statistics
    v_it = vt + chi_u_t(1,:);
    w_it = omegat + chi_u_t(2,:);
    th_itm1 = chi_x_tm1(3,:);
    motion = [...
        (-v_it./w_it).*sin(th_itm1) + (v_it./w_it).*sin(th_itm1 + w_it*Ts);...
        (v_it./w_it).*cos(th_itm1) - (v_it./w_it).*cos(th_itm1 + w_it*Ts);...
        w_it*Ts];
    chibar_x_t = chi_x_tm1 + motion; %sigma points propegated to t from t-1 (3x15)

    beta = 2;
    wm = [lamda/(n+lamda), 1/(2*(n+lamda))*ones(1,2*n)]; %weights for sigma points
    wc = [lamda/(n+lamda) + (1-alpha^2 + beta), 1/(2*(n+lamda))*ones(1,2*n)]; %weights for sigma points
    mu_bar_t = zeros(3,1);
    for i = 1:3
        mu_bar_t(i) = dot(wm,chibar_x_t(i,:)); %weighted mean of state sigma points
    end
    sigma_bar_t = zeros(3);
    for i = 1:2*L+1
        sigma_bar_t = sigma_bar_t + wc(i)*(chibar_x_t(:,i) - mu_bar_t)*(chibar_x_t(:,i) - mu_bar_t).'; % weighted covariance of state sigma points
    end
    
    %Predict observations at sigma points and compute gaussian statistics
    z_bar_t = [((mx - chibar_x_t(1,:)).^2 + (my - chibar_x_t(2,:)).^2).^0.5 + chi_z_t(1,:);...
        atan2(my - chibar_x_t(2,:), mx - chibar_x_t(1,:)) - chibar_x_t(3,:) + chi_z_t(2,:)];
    z_hat_t = zeros(2,1);
    for i = 1:2
        z_hat_t(i) = dot(wm,z_bar_t(i,:)); %weighted mean of measurement sigma points
    end
    St = zeros(2);
    for i = 1:2*L+1
        St = St + wc(i)*(z_bar_t(:,i) - z_hat_t)*(z_bar_t(:,i) - z_hat_t).';
    end
    sigma_xz_t = zeros(3,2); %cross covariance between state and measurement (3x2)
    for i = 1:2*L+1
       sigma_xz_t = sigma_xz_t + wc(i)*(chibar_x_t(:,i) - mu_bar_t)*(z_bar_t(:,i) - z_hat_t).';
    end
    
    %Update mean and covariance
    Kt = sigma_xz_t/St; %Kalman Gain
    mu_t = mu_bar_t + Kt*(z_t-z_hat_t); %measurement update
    sigma_t = sigma_bar_t - Kt*St*Kt.'; %update measurement covariance
end

