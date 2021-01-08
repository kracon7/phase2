img_folder = 'laser calibration/';
dlist = dir(img_folder);
[N, ~] = size(dlist);

% data cell structure:
% {theta, laser position array, all images}
data = cell(N-2, 3);
counter_d = 1;
for i = 1:N
    dname = dlist(i).name;
    if not(strcmp(dname, '.')) && not(strcmp(dname, '..'))
        fprintf('Working on data from dir %s\n', dname);
        % get theta
        theta = str2double(dname) / 7;
        
        dir_path = sprintf('%s%s', img_folder, dname);
        flist = dir(dir_path);
        [M, ~] = size(flist);
        
        % store data
        data(counter_d, 1) = {theta};
        
        positions = zeros(M-2, 3);
        images = zeros(M-2, 720, 1280, 3);
        
        counter_f = 1;
        for j = 1:M
           fname = flist(j).name;
           if not(strcmp(fname, '.')) && not(strcmp(fname, '..'))
               
               fpath = sprintf('%s/%s', flist(j).folder, flist(j).name);
               fprintf('Loading image from %s', fpath);
               
               % load image
               img = imread(fpath);
               
               % compute laser point position in camera frame
               positions(counter_f,:) = laser_position_in_camera(img);
               images(counter_f, :, :, :) = img;
               
               counter_f = counter_f +1;
           end
        end
        
        % store data
        data(counter_d, 2) = {positions};
        data(counter_d, 3) = {images};
        counter_d = counter_d + 1;
    end
end

% fit plane with all the points

% project all the points to the plane

% 