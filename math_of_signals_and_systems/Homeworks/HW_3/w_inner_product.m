function [ w_inner_prod ] = w_inner_product( t,f,g )
%INNER_PRODUCT Takes the numerical inner product of two functions
%   This is taking the integral defined on page 120 of the text in middle
%   of example 2.15.1
dt = diff(t);
prod = f.*g;
weighting = zeros(length(t),1);
for i = 1: length(t),
    weighting(i) = 1/sqrt(1-t(i)^2);
end

integrand = prod.*weighting;
area = 0;
for i = 1:length(dt)
    area = area + dt(i)*((integrand(i+1) + integrand(i))/2);
end
w_inner_prod = area;
end
