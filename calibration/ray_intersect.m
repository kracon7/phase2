function [position] = ray_intersect(T, center, k_intrisic_matrix)
    % do ray and plane intersection to get the intersection point 3d
    % position, the laser point center 3d position in the camera frame
    % Input:
    %   T -- transformation of checkerboard pattern in wrt camera frame
    %   center -- laser pointer center pixel value (px, py);  px point
    %               downward, py point right
    %             In the image, the camera frame x point right, y point
    %             down
    %   k_intrinsic_matrix -- camera intrinsic matrix
    
    % get the laser ray vector in camera frame
    ray = inv(k_intrisic_matrix) * [center(2), center(1), 1]';
    
    % get checker board origin and z axis in camera frame
    checker_origin = T(1:3, 4);
    checker_z = T(1:3, 3);
    
    % compute z of the laser point based on:  (z*ray - origin) .* checker_z = 0
    z = (checker_origin' * checker_z) / (ray' * checker_z);
    position = z * ray;
    
end


