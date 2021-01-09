function distance=point_line_dist(px, py, pz, v1, v2)

pt = [px, py, pz];
a = v1 - v2;
b = pt - v2;
distance = sum(sqrt(sum(cross(a,b,2).^2,2)) ./ sqrt(sum(a.^2,2)), 'all');
%this is equivalent to the following line for a single point
%distance=norm(cross(v1-v2,pt-v2))/norm(v1-v2)
end