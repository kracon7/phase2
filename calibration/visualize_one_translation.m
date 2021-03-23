function [] = visualize_results(data, r0_array, vec_array, origin, x_axis, y_axis, z_axis)
    [~,n_cells,~] = size(data);
    thetas = cell2mat(data(1, :, 1));
    for i = 1:n_cells
        points = cell2mat(data(1, i, 2));
        r0 = r0_array(i,:);
        vec = vec_array(i,:);
        
        % plot the lines and points
        scatter3(points(:,1), points(:,2), points(:,3), 'DisplayName',num2str(thetas(i)));
        hold on

        t = [-1500:5:1000];
        y = r0 + t' * vec;
        plot3(y(:,1), y(:,2), y(:,3));
        hold on
    end
    
    legend
    
    plot3(origin(1), origin(2), origin(3), 'o')    
    hold on
    
    scale = 200;
    quiver3(origin(1), origin(2), origin(3), scale*x_axis(1), scale*x_axis(2), scale*x_axis(3), 'r', 'lineWidth', 2)
    hold on
    quiver3(origin(1), origin(2), origin(3), scale*y_axis(1), scale*y_axis(2), scale*y_axis(3), 'g', 'lineWidth', 2)
    hold on
    quiver3(origin(1), origin(2), origin(3), scale*z_axis(1), scale*z_axis(2), scale*z_axis(3), 'b', 'lineWidth', 2)
    hold on
    
    % plot lines of different theta angles
    scale = 500;
     for i =1:n_cells
         theta = -thetas(i);
         R_theta = [1, 0, 0; 0, cosd(theta), -sind(theta); 0, sind(theta), cosd(theta)];
         R_cl = [x_axis', y_axis', z_axis'];
         R = R_cl * R_theta;
         z_rotated = R(:, 3);
         quiver3(origin(1), origin(2), origin(3), scale*z_rotated(1), scale*z_rotated(2), scale*z_rotated(3), 'k')
         hold on
     end
    

    xlim([-400, 400])
    ylim([-400, 600])
    zlim([-700, 2200])
    axis equal
    xlabel('x')
    ylabel('y')
    hold off
end
