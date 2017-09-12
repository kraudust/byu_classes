t = linspace(-1, 1, 10000);
p = zeros(10000, 5);
for kk = 1:5
    p(:, kk) = t.^(kk-1);
end
p1 = p(:,1);
p2 = p(:,2);
p3 = p(:,3);
p4 = p(:,4);
p5 = p(:,5);
induced_norm_p1 = inner_product(t,p1,p1).^0.5;
q1 = p1/induced_norm_p1;

q1_phil = p(:,1)/norm(p(:,1)); %Is this the right norm??
