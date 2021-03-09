function [I, center] = filter_laser( I1, visualize)
    [m,n,~] = size(I1);
    
    if visualize
        figure;
        subplot(1,2, 1)
        imshow(I1);
    end
    
    % rgb filter for light spot rought location
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

    
    % find spot rough size and location
    [O, R] = incircle(pixels(:, 1), pixels(:, 2));
    O = ceil(O);
    R = ceil(R);    
    
    patch = I1(O(1)-R : O(1)+R, O(2)-R : O(2)+R, :);
    brightness = sum(patch, 3);
    max_intensity = max(brightness, [], 'all');
    
    pixels = [];
    for i = 1 : 2*R+1
        for j = 1 : 2*R+1
            intensity = sum(patch(i,j,:), 'all');
            if intensity > 0.95 * max_intensity
                pixels = [pixels; [O(1)-R + i-1, O(2)-R + j-1]];
                I(O(1)-R + i-1, O(2)-R + j-1) = 2;
            end
        end
    end
    center = mean(pixels, 1);
    if visualize
        subplot(1,2,2)
        imshow(I, [0,2])
        hold on
        cx = O(1);
        cy = O(2);
        plot(cy, cx,'r','LineWidth',2);
        hold on
        theta = [linspace(0,2*pi) 0];
        plot(cos(theta)*R+cy,sin(theta)*R+cx,'color','g','LineWidth', 2);
        hold on
        plot(center(2), center(1), 'rx')
    end
end


