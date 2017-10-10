function [chibar_t] = low_variance_sampler(chi_t, w_t)
    % Table 4.4 Algorithm in Probabilistic Robotics
    w_t = w_t/sum(w_t);
    M = size(w_t, 2);
    r = (1/M)*rand();
    c = w_t(1);
    i = 1;
    chibar_t = zeros(size(chi_t));
    for m = 1:M
        U = r + (m - 1)*(1/M);
        while U > c
           i = i + 1;
           c = c + w_t(i);
        end
        chibar_t(:,m) = chi_t(:,i);
    end
end

