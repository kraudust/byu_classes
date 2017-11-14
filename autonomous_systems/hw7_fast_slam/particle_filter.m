function [chi_t, mu_t, sigma_t] = particle_filter(chi_tm1, u_tm1, z_t, lm, Ts, alpha, sigma_r, sigma_phi)
    %Table 8.2 Algorithm in Probabilistic Robotics
    xt_m = velocity_motion_model(u_tm1, chi_tm1, alpha, Ts);
    %covariance after propegation
    Q = [(std(xt_m(1,:),0,2))^2         0           0;...
        0                   (std(xt_m(2,:),0,2))^2          0;...
        0                   0               (std(xt_m(3,:),0,2))^2];
        
    wt_m = measurement_prob(z_t, xt_m, lm, sigma_r, sigma_phi);
    max(wt_m)
    wt_m = wt_m/sum(wt_m);
    chi_t = low_variance_sampler(xt_m, wt_m);
    
    % Calculate number of unique particles after resampling
    num_unique = size(unique(chi_t),2);
    mu_t = mean(chi_t,2); %mean of points
    if num_unique/size(chi_t,2) < 0.5 %if less than half the particles are unique
        Q = Q/((size(xt_m,2)*num_unique)^(1/size(xt_m,1)));
        chi_t = chi_t + mvnrnd(zeros(3,1), Q, size(xt_m,2)).';
    end
    
    sigma_t = [sqrt(Q(1,1));sqrt(Q(2,2));sqrt(Q(3,3))];
end

