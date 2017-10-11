function [q] = measurement_prob(z_t, xt_m, lm, sigma_r, sigma_phi)
    % z_t is size 2*num_landmarks x 1
    % xt_m is size 3 x num_particles
    % lm is size num_landmarks x 2
    q_lm = zeros(size(lm,1),size(xt_m,2));
    q = ones(1,size(xt_m,2));
    
    for i = 1:size(lm,1)
        rhat = ((lm(i,1) - xt_m(1,:)).^2 + (lm(i,2) - xt_m(2,:)).^2).^0.5;
        phihat = atan2(lm(i,2) - xt_m(2,:), lm(i,1) - xt_m(1,:)) - xt_m(3,:);
        q_lm(i,:) = pdf('Normal',z_t(i) - rhat, 0, sigma_r).*pdf('Normal',z_t(i+size(lm,1)) - phihat, 0, sigma_phi);
        q = q.*q_lm(i,:);
    end
end

