function sol=find_origin_from_lines(v1, v2)
    % find cross point of the lines given two points of each line
    % Input
    %   v1: N x 3
    %   v2: N x 3
    % Output
    %   sol: cross point
    
    func = @(pt) point_line_dist(pt(1), pt(2), pt(3), v1, v2);
    sol_guess = [10 20 30];
    sol = fminsearch(func, sol_guess);

end