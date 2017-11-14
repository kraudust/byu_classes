function idx = low_variance_sampler(w_t)
    % Table 4.4 Algorithm in Probabilistic Robotics
    idx = [];
    M = size(w_t, 2);
    r = (1/M)*rand();
    c = w_t(1);
    i = 1;
    for m = 1:M
        U = r + (m - 1)*(1/M);
        while U > c
           i = i + 1;
           c = c + w_t(i);
        end
        idx = [idx, i];
    end
end

