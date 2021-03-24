% calibrate;
% [all_projected_data, semi_calibrated_data] = compute_origin(data);
% visualize_all(all_projected_data, semi_calibrated_data);


figure

% plot camera
R = [1 0 0; 0 1 0;0 0 1];
t = [0 0 0];
pose = rigid3d(R,t);
cam = plotCamera('AbsolutePose',pose,'Opacity',0, 'Size',10);
hold on

axis equal
xlim([-200, 200])
ylim([-200, 300])
zlim([-300, 300])
xlabel('x')
ylabel('y')

[num_theta1, num_theta2, ~] = size(data);
thetas = cell2mat(all_projected_data(:, 1, 1)); 

origins = cell2mat(semi_calibrated_data(:, 3));
x_axis = cell2mat(semi_calibrated_data(:, 4));
y_axis = cell2mat(semi_calibrated_data(:, 5));
z_axis = cell2mat(semi_calibrated_data(:, 6));

r1_origin = find_origin_from_lines(origins, origins + x_axis);
r1_axis = find_axis_from_lines(x_axis);
r1_axis = r1_axis / norm(r1_axis);
r1_axis = -1 * sign(r1_axis(3)) * r1_axis;

visualize_all(all_projected_data, semi_calibrated_data);

plot3(r1_origin(1), r1_origin(2), r1_origin(3), 'mo');
hold on

scale = 100;
quiver3(r1_origin(1), r1_origin(2), r1_origin(3), scale*r1_axis(1), scale*r1_axis(2), scale*r1_axis(3), 'm', 'lineWidth', 2)
hold on