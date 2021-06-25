% calibrate the images and get the 3d positions of the laser positions in
% each pictures

clear
close all
img_folder = '../data/';
root_dir_list = dir(img_folder);
[num_theta1, ~] = size(root_dir_list);

visualize = 0;

n_rot_1 = 7;
n_rot_2 = 8;

% data cell structure:
% {theta, laser position array, color images, depth images, index}
data = cell(n_rot_1, n_rot_2, 5);

counter_theta1 = 1;

for i = 1:num_theta1
    theta1_dir_name = root_dir_list(i).name;
    if not(strcmp(theta1_dir_name, '.')) && not(strcmp(theta1_dir_name, '..'))
        fprintf('Working on data in translation dir %s\n', theta1_dir_name);
        
        % list theta_dir in trans_dir
        theta1_dir_path = sprintf('%s%s', img_folder, theta1_dir_name);
        theta1_dir_list = dir(theta1_dir_path);
        [num_theta2, ~] = size(theta1_dir_list);
        
        counter_theta2 = 1;
        
        for k = 1:num_theta2
            theta2_dir_name = theta1_dir_list(k).name;
            if not(strcmp(theta2_dir_name, '.')) && not(strcmp(theta2_dir_name, '..'))
                fprintf('Working on data in theta dir %s\n', theta2_dir_name);
                
                % get two rotation angles
                theta = [str2double(theta1_dir_name) / 3, str2double(theta2_dir_name) / 7];
                
                % list images in theta_dir
                theta2_dir_path = sprintf('%s%s/%s', img_folder, theta1_dir_name, theta2_dir_name);
                theta_dir_list = dir(theta2_dir_path);
                
                [num_points, ~] = size(theta_dir_list);
                fprintf('Found %d files from this dir\n', num_points-2);
                
                 % store data
                data(counter_theta1, counter_theta2, 1) = {theta};

                positions = [];
                color_images = [];
                depth_images = [];
                all_index = [];
                
                counter_f = 0;
                for j = 1:num_points
                    fname = theta_dir_list(j).name;
                    if not(strcmp(fname, '.')) && not(strcmp(fname, '..')) && contains(fname, '.png')
                        % get color image file name and find corresponding depth file name
                        color_fname = fname;
                        temp = split(convertCharsToStrings(fname), '.');
                        temp = split(temp(1), '_');
                        
                        depth_fname = sprintf('depth_%s.mat', temp(2));
                       
                        % load color image
                        color_fpath = sprintf('%s/%s', theta_dir_list(j).folder, color_fname);
                        fprintf('Loading image from %s\n', color_fpath);
                        color = imread(color_fpath);

                        % load depth image
                        depth_fpath = sprintf('%s/%s', theta_dir_list(j).folder, depth_fname);
                        fprintf('Loading image from %s\n', depth_fpath);
                        depth_strct = load(depth_fpath);
                        depth = depth_strct.depth;
                        
                        % compute laser point position in camera frame
                        pos = laser_position_in_camera(color, depth, visualize);
                        if pos(3) > 0                           
                            positions = [positions; pos];
                            color_images = [color_images; color];
                            depth_images = [depth_images; depth];
                            all_index = [all_index; str2double(temp(2))];
                            counter_f = counter_f +1;
                        else
                            fprintf('Invalid depth value! Skipped this point!\n');
                        end
                    end
                end
                % store data
                [im_h, im_w] = size(depth);
                data(counter_theta1, counter_theta2, 2) = {positions};
                data(counter_theta1, counter_theta2, 3) = {reshape(color_images, [], im_h, im_w, 3)};
                data(counter_theta1, counter_theta2, 4) = {reshape(depth_images, [], im_h, im_w)};
                data(counter_theta1, counter_theta2, 5) = {all_index};
                counter_theta2 = counter_theta2 + 1;
            end
            
        end
        
        counter_theta1 = counter_theta1 + 1;
    end
        
        
end