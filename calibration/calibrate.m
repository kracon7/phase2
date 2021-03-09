img_folder = '../data/';
root_dir_list = dir(img_folder);
[num_trans, ~] = size(root_dir_list);

visualize = 1;

% data cell structure:
% {theta, laser position array, color images, depth images, index}
data = cell(num_trans-2, 5, 5);

counter_trans = 1;

for i = 1:num_trans
    trans_dir_name = root_dir_list(i).name;
    if not(strcmp(trans_dir_name, '.')) && not(strcmp(trans_dir_name, '..'))
        fprintf('Working on data in translation dir %s\n', trans_dir_name);
        
        % list theta_dir in trans_dir
        trans_dir_path = sprintf('%s%s', img_folder, trans_dir_name);
        trans_dir_list = dir(trans_dir_path);
        [num_theta, ~] = size(trans_dir_list);
        
        counter_theta = 1;
        
        for k = 1:num_theta
            theta_dir_name = trans_dir_list(i).name;
            if not(strcmp(theta_dir_name, '.')) && not(strcmp(theta_dir_name, '..'))
                fprintf('Working on data in theta dir %s\n', theta_dir_name);
                
                % get theta
                theta = str2double(trans_dir_name) / 7;
                
                % list images in theta_dir
                theta_dir_path = sprintf('%s%s/%s', img_folder, trans_dir_name, theta_dir_name);
                theta_dir_list = dir(theta_dir_path);
                
                [num_points, ~] = size(theta_dir_list);
                fprintf('Found %d files from this dir\n', num_points-2);
                
                 % store data
                data(counter_trans, counter_theta, 1) = {theta};

                positions = zeros((num_points-2)/2, 3);
                color_images = zeros((num_points-2)/2, 720, 1280, 3);
                depth_images = zeros((num_points-2)/2, 720, 1280);
                all_index = zeros((num_points-2)/2, 1);
                
                counter_f = 1;
                for j = 1:num_points
                    fname = theta_dir_list(j).name;
                    if not(strcmp(fname, '.')) && not(strcmp(fname, '..')) && contains(fname, '.png')
                        % get color image file name and find corresponding depth file name
                        color_fname = fname;
                        temp = split(convertCharsToStrings(fname), '.');
                        temp = split(temp(1), '_');
                        all_index(counter_f, 1) = str2double(temp(2));
                        depth_fname = sprintf('depth_%s.mat', temp(2));
                       
                        % load color image
                        color_fpath = sprintf('%s/%s', theta_dir_list(j).folder, color_fname);
                        fprintf('Loading image from %s\n', color_fpath);
                        color = imread(color_fpath);

                        % load depth image
                        depth_fpath = sprintf('%s/%s', theta_dir_list(j).folder, depth_fname);
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
                data(counter_trans, counter_theta, 2) = {positions};
                data(counter_trans, counter_theta, 3) = {color_images};
                data(counter_trans, counter_theta, 4) = {depth_images};
                data(counter_trans, counter_theta, 5) = {all_index};
                counter_theta = counter_theta + 1;
            end
            
        end
        
        counter_trans = counter_trans + 1;
    end
        
        
end
        
        
        
        
       
        
        
           
               
               
               
              
               
               
          
        

% fit plane with all the points

% project all the points to the plane

% 