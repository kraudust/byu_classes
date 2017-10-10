function [chi_t] = particle_filter(chi_tm1, u_tm1, z_t, lm, Ts, alpha, sigma_r, sigma_phi)
    %Table 8.2 Algorithm in Probabilistic Robotics
    xt_m = velocity_motion_model(u_tm1, chi_tm1, alpha, Ts);
    wt_m = measurement_prob(z_t, xt_m, lm, sigma_r, sigma_phi);
    
    chi_t = low_variance_sampler(xt_m, wt_m);
    
end

