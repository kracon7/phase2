function [T1] = inv_transformation( T0)
    % inverse the transformation
    R = T0(1:3, 1:3);
    p = T0(1:3, 4);
    T1 = [R'; -R' * p];
end


