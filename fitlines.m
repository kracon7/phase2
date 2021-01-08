function [r0, vec] = fitlines(points_cells, visualize)
    % fit lines to the given cells of 3d points
    % Input:
    %   points_cells -- cell array of 3d point arrays of shape N x 3
    if visualize
        figure
    end
    
    [n_cells,~] = size(points_cells);
    for i = 1:n_cells
        points = cell2mat(points_cells(i, 1));
        [r0, vec] = fit_one_line(points);
        
        if visualize
            % plot the lines and points
            scatter3(points(:,1), points(:,2), points(:,3));
            hold on

            t = [-1:0.05:1];
            y = r0 + t' * vec';
            plot3(y(:,1), y(:,2), y(:,3));
            hold on
        end
    end
    
    hold off
end


