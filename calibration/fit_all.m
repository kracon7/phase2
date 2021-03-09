% calibrate;
% [all_projected_data, semi_calibrated_data] = compute_origin(data);
% visualize_all(all_projected_data, semi_calibrated_data);

all_origins = cell2mat(semi_calibrated_data(:,3));

[r0, vec] = fit_one_line(all_origins(2:end,:));
