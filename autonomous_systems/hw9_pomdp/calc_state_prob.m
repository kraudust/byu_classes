function p = calc_state_prob(xi, u, xj)
% calculates p(xi|u,xj)
    if u == 1 || u == 2
       p = 0; 
    else % action is u3
        if xi == xj
            p = 0.2;
        else
            p = 0.8;
        end
    end
end

