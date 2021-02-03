function [position] = laser_position_in_camera(color, depth, visualize)
    % compute the laser point center 3d position in the camera frame 
    % Input:
    %   img -- image of the checker board and laser point
    
    k_intrisic_matrix=[918.2091437945788 0.0 639.193207006314;
                    0.0   921.9954810539982    358.461790471607;
                    0.0    0.0    1.0];
    
    [~, center] = filter_laser(color, visualize);
    
    if visualize
        imshow(depth);
    end
    
     % get the laser ray vector in camera frame
    ray = inv(k_intrisic_matrix) * [center(2), center(1), 1]';
    
    z = depth(floor(center(1)), floor(center(2)));
    
    position = double(z) * ray;
end


