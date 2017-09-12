function [ inner_prod ] = inner_product( t,f,g )
%INNER_PRODUCT Takes the numerical inner product of two functions
%   This is taking the integral defined on page 120 of the text at the top
%   of example 2.15.1
dt = diff(t);
prod = f.*g;
area = 0;
for i = 1:length(dt)
    area = area + dt(i)*((prod(i+1) + prod(i))/2);
end
inner_prod = area;
end

