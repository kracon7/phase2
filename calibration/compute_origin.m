function [all_projected_data, semi_calibrated_data] = compute_origin(data)
    % first project points to a plane and then fit the lines given 3d 
    % points, compute the origins and frame axis
    % Input:
    %   data -- data cells of position information
    
    
    [num_theta1, num_theta2, ~] = size(data);
    
    visualize_fitplane = 0;
    visualize_fitlines = 0;
    
    all_projected_data = cell(size(data));
    semi_calibrated_data = cell(num_theta1, 6);
    
    for i = 1:num_theta1
        thetas = reshape(cell2mat(data(i, :, 1)), 2, num_theta2)';  
        thetas = thetas(:, 2);

        % compute fitted plane parameters
        B = fitplane(data(i,:,2), visualize_fitplane);
        a = B(1);
        b = B(2);
        c = B(3);
        % B for plane parameters -- ax + by + c = z
        x_axis = [a, b, -1];
        x_axis = sign(x_axis(1)) * x_axis / norm(x_axis);
        

        % project all points to the plane
        projected_data = data(i, :, :);

        for j = 1:num_theta2
            points = cell2mat(data(i, j, 2));
            Z = points(:,3);
            Y = points(:,2);
            X = (Z - b*Y - c* ones(size(Z))) / a;
            anchor = [X, Y, Z];
            v = points - anchor;
            w = [a, b, -1];
            projected = points - v * w' * w/ norm(w)^2;

            projected_data(1, j, 2) = {projected};
        end

        [r0, vec] = fit_lines(projected_data, visualize_fitlines);

        origin = find_origin_from_lines(r0, r0+vec);

        % find x axis, which is the line of theta = 0 degree
        for j =1:num_theta2
            theta = thetas(j);
            if theta == 0
                z_axis = - vec(j, :);
                z_axis = sign(z_axis(3)) * z_axis / norm(z_axis);
                break
            end
        end

        y_axis = cross(z_axis, x_axis);

        visualize_one_translation(projected_data, r0, vec, origin, x_axis, y_axis, z_axis);
        
        all_projected_data(i, :, :) = projected_data;
        semi_calibrated_data(i, 1) = {r0};
        semi_calibrated_data(i, 2) = {vec};
        semi_calibrated_data(i, 3) = {origin};
        semi_calibrated_data(i, 4) = {x_axis};
        semi_calibrated_data(i, 5) = {y_axis};
        semi_calibrated_data(i, 6) = {z_axis};
    end
end

