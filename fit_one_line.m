function [r0, vec] = fit_one_line(points)
    % fit one line to the given 3d points
    % Input:
    %   points -- 3d point arrays of shape N x 3
    
    r0=mean(points);
    points = bsxfun(@minus,points,r0);
    [~,~,V] = svd(points,0);
    vec = V(:, 1);
%     r(t) = r0 + t * V(:,1);
    
end


