img_folder = 'laser_calibration/';
dlist = dir(img_folder);
[N, ~] = size(dlist);

visualize = 0;

% data cell structure:
% {theta, laser position array, color images, depth images, index}
data = cell(N-2, 5);
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
        fprintf('Found %d files from this dir\n', M-2);
        
        % store data
        data(counter_d, 1) = {theta};
        
        positions = zeros((M-2)/2, 3);
        color_images = zeros((M-2)/2, 720, 1280, 3);
        depth_images = zeros((M-2)/2, 720, 1280);
        all_index = zeros((M-2)/2, 1);
        
        counter_f = 1;
        for j = 1:M
           fname = flist(j).name;
           if not(strcmp(fname, '.')) && not(strcmp(fname, '..')) && contains(fname, '.png')
               
               % get color image file name and find corresponding depth file name
               color_fname = fname;
               temp = split(convertCharsToStrings(fname), '.');
               temp = split(temp(1), '_');
               all_index(counter_f, 1) = str2double(temp(2));
               depth_fname = sprintf('depth_%s.mat', temp(2));
               
               % load color image
               color_fpath = sprintf('%s/%s', flist(j).folder, color_fname);
               fprintf('Loading image from %s\n', color_fpath);
               color = imread(color_fpath);
               
               % load depth image
               depth_fpath = sprintf('%s/%s', flist(j).folder, depth_fname);
               fprintf('Loading image from %s\n', depth_fpath);
               temp = load(depth_fpath);
               depth = temp.depth;
               
               % compute laser point position in camera frame
               positions(counter_f,:) = laser_position_in_camera(color, depth, visualize);
               pause(0.5)
               color_images(counter_f, :, :, :) = color;
               depth_images(counter_f, :, :) = depth;
               
               counter_f = counter_f +1;
           end
        end
        
        % store data
        data(counter_d, 2) = {positions};
        data(counter_d, 3) = {color_images};
        data(counter_d, 4) = {depth_images};
        data(counter_d, 5) = {all_index};
        counter_d = counter_d + 1;
    end
end

% fit plane with all the points

% project all the points to the plane

% 