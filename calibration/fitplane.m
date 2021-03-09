function [B] = fitplane(points_cells, visualize)
    % fit lines to the given cells of 3d points
    % Input:
    %   points_cells -- cell array of 3d point arrays of shape N x 3
    % Output:
    %   B -- ax + by + c = z
    if visualize
        figure
    end
    
    [~, n_cells] = size(points_cells);
    points = [];
    for i = 1:n_cells
        points = [points; cell2mat(points_cells(1, i))];
    end
    
    x=points(:,1);
    y=points(:,2);
    z=points(:,3);
    DM = [x, y, ones(size(z))];                             % Design Matrix
    B = DM\z;                                               % Estimate Parameters
    [X,Y] = meshgrid(linspace(min(x),max(x),50), linspace(min(y),max(y),50));
    Z = B(1)*X + B(2)*Y + B(3)*ones(size(X));
    plot3(x, y, z, '.')
    hold on
    meshc(X, Y, Z)
    hold off
    grid on
    xlabel('x(mm)'); ylabel('y(mm)'); zlabel('z(mm)');
    title('Masked plot');
    grid on
    
    xlim([-400, 400])
    ylim([-300, 500])
    zlim([-200, 2200])
    xlabel('x')
    ylabel('y')
    hold off
end


