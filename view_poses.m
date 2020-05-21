clear all;
close all;

% points3D_start = importdata('points3D_start.txt');
% points3D_start = points3D_start(:,1:3);

figure
pcshow([0,0,0],'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on

minus_cameras = 0;

% gt_cam_loc = [];
dinfo = dir('poses/global_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses/global_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
%     gt_cam_loc = [gt_cam_loc ; camera_location'];
    
    % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [1, 0, 0]);
end

hold on

% fp_cam_loc = [];
dinfo = dir('poses/ar_core_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses/ar_core_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
%     fp_cam_loc = [fp_cam_loc ; camera_location'];

    % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [0, 1, 0]);
end

hold on

% fp_cam_loc = [];
dinfo = dir('poses/final_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses/final_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
%     fp_cam_loc = [fp_cam_loc ; camera_location'];

    % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [0, 0, 1]);
end


