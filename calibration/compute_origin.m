function [origin, x_axis, y_axis, z_axis] = compute_origin(data)
    % fit one line to the given 3d points using ransac
    % Input:
    %   data -- data cells of position information
    
    thetas = cell2mat(data(:, 1));
    
    % compute fitted plane parameters
    B = fitplane(data(:,2), 1);
    a = B(1);
    b = B(2);
    c = B(3);
    % B for plane parameters -- ax + by + c = z
    z_axis = [a, b, -1];
    z_axis = - z_axis / norm(z_axis);
    
    % project all points to the plane
    projected_data = data;
    [n_cells,~] = size(data);
    for i = 1:n_cells
        points = cell2mat(data(i, 2));
        Z = points(:,3);
        Y = points(:,2);
        X = (Z - b*Y - c* ones(size(Z))) / a;
        anchor = [X, Y, Z];
        v = points - anchor;
        w = [a, b, -1];
        projected = points - v * w' * w/ norm(w)^2;
        
        projected_data(i, 2) = {projected};
    end
    
%     B_p = fitplane(projected_data(:,2), 1);
    
    [r0, vec] = fit_lines(projected_data, 1);
    
    origin = find_origin_from_lines(r0, r0+vec);
    
    % find x axis, which is the line of theta = 0 degree
    for i =1:n_cells
        theta = thetas(i);
        if theta == 0
            x_axis = vec(i, :);
            x_axis = x_axis / norm(x_axis);
            break
        end
    end
    
    y_axis = cross(z_axis, x_axis);
    
    
    visualize_results(projected_data, r0, vec, origin, x_axis, y_axis, z_axis);
end

