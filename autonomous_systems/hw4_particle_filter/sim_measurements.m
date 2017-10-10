function [r, phi] = sim_measurements(m,sigma_r, sigma_phi,xt)
    % Table 6.4 Algorithm in Probabilistic Robotics
    %size of m is #landmarks by 2 (x & y location of each landmark)
    r = zeros(size(m,1),size(xt,2));
    phi = zeros(size(m,1),size(xt,2));
    for i = 1:size(xt,2)
        for j = 1:size(m,1)
            r(j,i) = sqrt((m(j,1)-xt(1,i))^2 + (m(j,2)-xt(2,i))^2) + sigma_r*randn();
            phi(j,i) = atan2(m(j,2)-xt(2,i), m(j,1)-xt(1,i)) - xt(3,i) + sigma_phi*randn();
        end
    end
end

