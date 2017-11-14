function Y_t = fastSLAM(z_t, u_tm1, Y_tm1)
    global landmarks_seen sigma_r sigma_phi alpha Ts
    M = length(Y_tm1); %number of particles
    for k = 1 to M
        x_tm1 =  % particle k's state [x;y;theta] at previous time step
    end
end

