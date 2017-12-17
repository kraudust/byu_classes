function r = reward(xi, u)
    if u == 3
        r = -1;        
    else
        if xi == 1
            if u == 1
                r = -100;
            else
                r = 100;
            end
        else
            if u == 1
                r = 100;
            else
                r = -50;
            end
        end
    end
end

