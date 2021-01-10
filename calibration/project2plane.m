function [projected] = project2plane(points, B)
    a = B(1);
    b = B(2);
    c = B(3);
    anchor = [0, 0, c];
    v = points - anchor;
    w = [a, b, -1];
    projected = points - v * w' * w/ norm(w)^2;
    
end
