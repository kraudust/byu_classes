function [chi_t] = particle_filter(chi_tm1, u_tm1, z_t, lm, Ts, alpha, sigma_r, sigma_phi, x_t)
    %Table 8.2 Algorithm in Probabilistic Robotics
    xt_m = velocity_motion_model(u_tm1, chi_tm1, alpha, Ts);
%     scatter(xt_m(1,:), xt_m(2,:))
%     hold on
%     scatter(x_t(1), x_t(2))
%     scatter(lm(:,1), lm(:,2))
    wt_m = measurement_prob(z_t, xt_m, lm, sigma_r, sigma_phi);
    wt_m = wt_m/sum(wt_m);
    chi_t = low_variance_sampler(xt_m, wt_m);
    %chi_t = xt_m;
    
end

