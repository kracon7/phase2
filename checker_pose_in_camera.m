function [T] = checker_pose_in_camera(I1, k_intrisic_matrix, visualize)
    [points, boardsize]= detectCheckerboardPoints(I1);
    grid_size=20;
    if visualize
        figure;
        imshow(I1);
        hold on;
        c=boardsize(1)-1;
        d=boardsize(2)-1;  
        x_axis_x=[points(1,1), points(c,1)];
        x_axis_y=[points(1,2), points(c,2)];
        hold on;
        plot(x_axis_x,x_axis_y,'g');

        y_axis_x=[points(1,1), points(c*(d-1)+1,1)];
        y_axis_y=[points(1,2), points(c*(d-1)+1,2)];
        hold on;
        plot(y_axis_x,y_axis_y,'b');
    end
    
    T = get__transformation_matrix(points, boardsize,grid_size,k_intrisic_matrix);
    T(1:3,4)=T(1:3,4)/1000;

end


