function chi_t = velocity_motion_model(u_tmin1, chi_tmin1, alpha, Ts)
    %see table 5.3 in Probabilistic Robotics
    v = u_tmin1(1);
    omega = u_tmin1(2);
    x = chi_tmin1(1, :);
    y = chi_tmin1(2, :);
    th = chi_tmin1(3, :);
    v_hat = v + sqrt(alpha(1)*v^2 + alpha(2)*omega^2)*randn(1,size(chi_tmin1,2));
    omega_hat = omega + sqrt(alpha(3)*v^2 + alpha(4)*omega^2)*randn(1,size(chi_tmin1,2));
    chi_t = [x - (v_hat./omega_hat).*sin(th) + (v_hat./omega_hat).*sin(th + omega_hat.*Ts);...
        y + (v_hat./omega_hat).*cos(th) - (v_hat./omega_hat).*cos(th + omega_hat.*Ts);...
        th + omega_hat.*Ts];
end

