function [I, center] = filter_laser( I1, visualize)
    [m,n,~] = size(I1);
    
    if visualize
        figure;
        subplot(1,2, 1)
        imshow(I1);
    end
    
    % rgb filter
    % 250, 192, 35
    % 227, 106, 70
    % 226, 113, 87
    % 233, 101, 16
    % 253, 142, 16
    % 220, 104, 62
    % 238, 106, 49
    % 253, 127, 38
    % 233, 103, 52
    filter_low = [220, 100, 10];
    filter_high = [256, 195, 90];
    
    I = zeros(m, n);
    pixels = [];
    for i = 1:m
        for j = 1:n
            pixel = I1(i,j,:);
            if pixel(1)>filter_low(1) && pixel(2)>filter_low(2) && pixel(3)>filter_low(3) &&...
                            pixel(1)<filter_high(1) && pixel(2)<filter_high(2) && pixel(3)<filter_high(3)
                I(i,j)=1;
                pixels = [pixels; [i, j]];
            end
        end
    end
    
    if visualize
        subplot(1,2,2)
        imshow(I)
        hold on
    end
    
    
    [center, R] = incircle(pixels(:, 1), pixels(:, 2));
    if visualize
        cx = center(1);
        cy = center(2);
        plot(cy, cx,'r','LineWidth',2);
        theta = [linspace(0,2*pi) 0];
        hold on
        plot(cos(theta)*R+cy,sin(theta)*R+cx,'color','g','LineWidth', 2);
    end
end


