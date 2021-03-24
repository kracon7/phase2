function sol=find_axis_from_lines(v)
    % find normal of the plane from lines given two points of each line
    % Input
    %   v: N x 3
    % Output
    %   sol: surface normal
    
    func = @(pt) inner_product(pt(1), pt(2), pt(3), v);
    sol_guess = [10 20 30];
    sol = fminsearch(func, sol_guess);

end