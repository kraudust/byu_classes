function p = calc_meas_prob(z, xi)
global meas_cor_prob
%calc probability of measurement z given state x
    if z == xi
        p = meas_cor_prob; 
    else
        p = 1-meas_cor_prob;
    end
end

