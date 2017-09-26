function xt = velocity_motion_model(u_tmin1, x_tmin1, alpha, Ts)
    %see table 5.3 in Probabilistic Robotics
    v = u_tmin1(1);
    omega = u_tmin1(2);
    x = x_tmin1(1);
    y = x_tmin1(2);
    th = x_tmin1(3);
    v_hat = v + sqrt(alpha(1)*v^2 + alpha(2)*omega^2)*randn();
    omega_hat = omega + sqrt(alpha(3)*v^2 + alpha(4)*omega^2)*randn();
    xt = [x - (v_hat/omega_hat)*sin(th) + (v_hat/omega_hat)*sin(th + omega_hat*Ts);...
        y + (v_hat/omega_hat)*cos(th) - (v_hat/omega_hat)*cos(th + omega_hat*Ts);...
        th + omega_hat*Ts];
end

