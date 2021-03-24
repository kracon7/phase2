function [r0_array, vec_array] = fit_lines(data, visualize)
    % fit lines to the given cells of 3d points
    % Input:
    %   points_cells -- cell array of 3d point arrays of shape N x 3
    
    thetas = cell2mat(data(1, :, 1));
    points_cells = data(1, :, 2);
    
    if visualize
        figure
    end
    
    r0_array = [];
    vec_array = [];
    
    [~,n_cells,~] = size(points_cells);
    for i = 1:n_cells
        points = cell2mat(points_cells(1, i, 1));
        [r0, vec] = fit_one_line(points);
        
        r0_array = [r0_array; r0];
        vec_array = [vec_array; vec];
        
        if visualize
            % plot the lines and points
            scatter3(points(:,1), points(:,2), points(:,3), 'DisplayName',num2str(thetas(i)));
            hold on

            t = [-1500:5:1000];
            y = r0 + t' * vec;
            plot3(y(:,1), y(:,2), y(:,3));
            hold on
        end
    end
    
    if visualize
        xlim([-1000, 1000])
        ylim([-600, 1200])
        zlim([-200, 2200])
        xlabel('x')
        ylabel('y')
        legend
        hold off
    end
end


