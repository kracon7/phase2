function [position] = laser_position_in_camera(img)
    % compute the laser point center 3d position in the camera frame 
    % Input:
    %   img -- image of the checker board and laser point
    
    k_intrisic_matrix=[918.2091437945788 0.0 639.193207006314;
                    0.0   921.9954810539982    358.461790471607;
                    0.0    0.0    1.0];
    
    T = checker_pose_in_camera(img, k_intrisic_matrix,  0);
    pause(0.5)
    [~, center] = filter_laser(img, 0);
    position = ray_intersect(T, center, k_intrisic_matrix);
end


